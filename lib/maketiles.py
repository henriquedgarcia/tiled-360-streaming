from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.logger import Logger
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.status_ctx import StatusCtx
from lib.assets.worker import Worker
from lib.utils.worker_utils import print_error, decode_video, run_command
from lib.utils.context_utils import task


class CheckTiles(CtxInterface):
    logger: Logger
    status: StatusCtx
    segmenter_paths: SegmenterPaths
    tiling_list: list

    def check_tile(self, decode_check=False):
        try:
            self.check_tile_log()
            self.check_tile_video(decode_check)
            raise AbortError('Tiles are Ok')
        except FileNotFoundError:
            self.clean_tile()

    def check_tile_log(self):
        try:
            compressed_log_text = self.segmenter_paths.tile_log.read_text()
        except FileNotFoundError:
            raise FileNotFoundError(f'Tile log not exist.')

        if 'encoded 1800 frames' not in compressed_log_text:
            self.logger.register_log('Tile log is corrupt', self.segmenter_paths.tile_log)
            raise FileNotFoundError('Tile log is corrupt')

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.logger.register_log('Codec version is different.', self.segmenter_paths.tile_log)
            raise FileNotFoundError('Codec version is different.')

    def check_tile_video(self, test_decoding):
        try:
            compressed_file_size = self.segmenter_paths.tile_video.stat().st_size
        except FileNotFoundError:
            raise FileNotFoundError(f'Tile not exist.')

        if compressed_file_size == 0:
            raise FileNotFoundError('Tile size == 0.')

        if test_decoding:
            print(f'\r\t\tDecoding check tile{self.tile}.', end='')
            self.decode_check()

    def decode_check(self):
        if self.status.get_status('decode_check'):
            print_error(f'. OK')
            return

        stdout = decode_video(self.segmenter_paths.tile_video)
        if "frame= 1800" not in stdout:
            self.logger.register_log(f'Decode tile error.', self.segmenter_paths.tile_video)
            raise FileNotFoundError('Decoding Compress Error')
        self.status.update_status('decode_check', True)
        print_error(f'. OK')

    def assert_lossless(self):
        if not self.segmenter_paths.lossless_video.exists():
            self.logger.register_log('lossless_video not found.', self.segmenter_paths.lossless_video)
            raise AbortError(f'lossless_video not found.')

    def clean_tile(self):
        self.segmenter_paths.tile_log.unlink(missing_ok=True)
        self.segmenter_paths.tile_video.unlink(missing_ok=True)


class MakeTiles(Worker, CheckTiles):
    quality_list: list[str] = None

    def main(self):
        self.segmenter_paths = SegmenterPaths(self.ctx)
        self.quality_list = ['0'] + self.ctx.quality_list
        self.make_tiles(decode_check=False)

    def iterate_name_projection_quality_tiling_tile(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.tile in self.tile_list:
                            self.ctx.iterations += 1
                            yield

    def make_tiles(self, decode_check):
        for _ in self.iterate_name_projection_quality_tiling_tile():
            print(f'==== MakeTiles {self.ctx} ====')
            try:
                self.make_tile(decode_check)
            except AbortError as e:
                print_error(f'\t{e.args[0]}')

    def make_tile(self, decode_check):
        print(f'\tChecking tiles')
        self.check_tile(decode_check)
        print(f'\tChecking lossless')
        self.assert_lossless()

        print(f'\tCreating Tiles ')
        cmd = self.make_tile_cmd()
        print('\t' + cmd)
        run_command(cmd, self.segmenter_paths.tile_folder, self.segmenter_paths.tile_log)

        self.check_tile(decode_check)
        self.logger.register_log('Tile Creation Error.', self.segmenter_paths.tile_video)

    def make_tile_cmd(self):
        lossless_file = self.segmenter_paths.lossless_video.as_posix()
        compressed_file = self.segmenter_paths.tile_video.as_posix()

        x1, y1, x2, y2 = self.tile_position_dict[self.scale][self.tiling][self.tile]

        gop_options = f'keyint={self.gop}:min-keyint={self.gop}:open-gop=0'
        misc_options = f'scenecut=0:info=0'
        qp_options = ':ipratio=1:pbratio=1' if self.rate_control == 'qp' else ''
        lossless_option = ':lossless=1' if self.quality == '0' else ''
        codec_params = f'-x265-params {gop_options}:{misc_options}{qp_options}{lossless_option}'
        codec = f'-x265-params {gop_options}:{misc_options}{qp_options}{lossless_option}'
        crop_params = f'crop=w={x2 - x1}:h={y2 - y1}:x={x1}:y={y1}'
        output_options = f'-{self.rate_control} {self.quality} -tune psnr'

        cmd = ('bash -c '
               '"'
               'bin/ffmpeg -hide_banner -y -psnr '
               f'-i {lossless_file} '
               f'{output_options} '
               f'{codec} {codec_params} '
               f'-vf {crop_params} '
               f'{compressed_file}'
               f'"')

        return cmd
