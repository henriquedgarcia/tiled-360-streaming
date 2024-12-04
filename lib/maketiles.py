from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.maketilespaths import MakeTilesPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import print_error, decode_video, run_command


class MakeTiles(Worker, CtxInterface):
    make_tiles_paths: MakeTilesPaths
    quality_list: list[str] = None
    decode_check = False

    def init(self):
        self.make_tiles_paths = MakeTilesPaths(self.ctx)
        self.quality_list = ['0'] + self.ctx.quality_list

    def main(self):
        self.init()
        for _ in self.iterate_name_projection_tiling_tile_quality():
            with task(self):
                self.make_tile()

    def make_tile(self):
        if self.tiles_is_ok():
            return 'All OK'

        cmd = self.make_tile_cmd()
        run_command(cmd=cmd,
                    folder=self.tile_folder,
                    log_file=self.tile_log,
                    ui_prefix='\t')

    def make_tile_cmd(self) -> str:
        lossless_file = self.lossless_video.as_posix()
        compressed_file = self.tile_video.as_posix()

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

    def tiles_is_ok(self) -> bool:
        print(f'\tChecking tiles')
        try:
            self.check_tile_log()
            self.check_tile_video()
            raise AbortError('Tiles are OK')
        except FileNotFoundError:
            self.clean_tile()

        try:
            self.assert_lossless()
        except FileNotFoundError:
            raise AbortError('Lossless not found.')

        return False

    def check_tile_log(self):
        compressed_log_text = self.tile_log.read_text()

        if 'encoded 1800 frames' not in compressed_log_text:
            self.logger.register_log('Tile log is corrupt', self.tile_log)
            self.tile_log.unlink()
            raise FileNotFoundError

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.logger.register_log('Codec version is different.', self.tile_log)
            self.tile_log.unlink(missing_ok=True)
            raise FileNotFoundError

    def check_tile_video(self):
        compressed_file_size = self.tile_video.stat().st_size

        if compressed_file_size == 0:
            raise FileNotFoundError('Filesize is 0')

        if self.decode_check:
            print(f'\r\t\tDecoding check tile{self.tile}.', end='')
            self.decode_tile()

        return True

    def decode_tile(self) -> None:
        if self.status.get_status('decode_check'):
            print_error(f'. OK')
            return

        stdout = decode_video(self.tile_video, ui_prefix='\t')
        if "frame= 1800" not in stdout:
            self.logger.register_log(f'Decode tile error.', self.tile_video)
            raise FileNotFoundError('Decode Tile Error')
        self.status.update_status('decode_check', True)
        print_error(f'. OK')

    def assert_lossless(self) -> None:
        print(f'\tChecking lossless')
        if not self.lossless_video.exists():
            self.logger.register_log('lossless_video not found.', self.lossless_video)
            raise AbortError(f'lossless_video not found.')

    def clean_tile(self) -> None:
        self.tile_log.unlink(missing_ok=True)
        self.tile_video.unlink(missing_ok=True)

    @property
    def tile_folder(self):
        return self.make_tiles_paths.tile_folder

    @property
    def tile_log(self):
        return self.make_tiles_paths.tile_log

    @property
    def tile_video(self):
        return self.make_tiles_paths.tile_video

    @property
    def lossless_video(self):
        return self.make_tiles_paths.lossless_video
