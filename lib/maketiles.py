from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.maketilespaths import MakeTilesPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.worker_utils import print_error, decode_video, run_command
from lib.utils.context_utils import task


class MakeTiles(Worker, CtxInterface):
    make_tiles_paths: MakeTilesPaths
    quality_list: list[str] = None
    decode_check = False

    def init(self):
        self.make_tiles_paths = MakeTilesPaths(self.ctx)
        self.quality_list = ['0'] + self.ctx.quality_list

    def main(self):
        self.init()
        for _ in self.iterate_name_projection_quality_tiling_tile():
            with task(self):
                self.make_tile()

    def make_tile(self):
        if self.tile_is_ok(): return
        if not self.lossless_is_ok():
            raise AbortError('Lossless is not ok')

        cmd = self.make_tile_cmd()
        run_command(cmd, self.make_tiles_paths.tile_folder,
                    self.make_tiles_paths.tile_log,
                    ui_prefix='\t')

    def make_tile_cmd(self):
        lossless_file = self.make_tiles_paths.lossless_video.as_posix()
        compressed_file = self.make_tiles_paths.tile_video.as_posix()

        x1, y1, x2, y2 = self.tile_position_dict[self.scale][self.tiling][self.tile]

        gop_options = f'keyint={self.gop}:min-keyint={self.gop}:open-gop=0'
        misc_options = f'scenecut=0:info=0'
        qp_options = ':ipratio=1:pbratio=1' if self.rate_control == 'qp' else ''
        lossless_option = ':lossless=1' if self.quality == '0' else ''
        codec_params = f'-x265-params {gop_options}:{misc_options}{qp_options}{lossless_option}'
        codec = f'-c:v libx265'
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

    def tile_is_ok(self):
        print(f'\tChecking tiles')
        if not (self.tile_log_is_ok()
                and self.tile_video_is_ok()):
            self.clean_tile()
            return False
        return True

    def tile_log_is_ok(self):
        try:
            compressed_log_text = self.make_tiles_paths.tile_log.read_text()
        except FileNotFoundError:
            return False

        if 'encoded 1800 frames' not in compressed_log_text:
            self.logger.register_log('Tile log is corrupt', self.make_tiles_paths.tile_log)
            self.make_tiles_paths.tile_log.unlink()
            return False

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.logger.register_log('Codec version is different.', self.make_tiles_paths.tile_log)
            self.make_tiles_paths.tile_log.unlink(missing_ok=True)
            return False
        return True

    def tile_video_is_ok(self):
        try:
            compressed_file_size = self.make_tiles_paths.tile_video.stat().st_size
        except FileNotFoundError:
            return False

        if compressed_file_size == 0:
            return False

        if self.decode_check:
            print(f'\r\t\tDecoding check tile{self.tile}.', end='')
            try:
                self.decode_tile()
            except FileNotFoundError as e:
                self.clean_tile()
                print_error('\t', e.args[0])
                return False

        return True

    def decode_tile(self):
        if self.status.get_status('decode_check'):
            print_error(f'. OK')
            return

        stdout = decode_video(self.make_tiles_paths.tile_video, ui_prefix='\t')
        if "frame= 1800" not in stdout:
            self.logger.register_log(f'Decode tile error.', self.make_tiles_paths.tile_video)
            raise FileNotFoundError('Decode Tile Error')
        self.status.update_status('decode_check', True)
        print_error(f'. OK')

    def lossless_is_ok(self):
        print(f'\tChecking lossless')
        if not self.make_tiles_paths.lossless_video.exists():
            self.logger.register_log('lossless_video not found.', self.make_tiles_paths.lossless_video)
            raise AbortError(f'lossless_video not found.')

    def clean_tile(self):
        self.make_tiles_paths.tile_log.unlink(missing_ok=True)
        self.make_tiles_paths.tile_video.unlink(missing_ok=True)
