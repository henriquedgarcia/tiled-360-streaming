from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.maketilespaths import MakeTilesPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import print_error, decode_video, run_command


class MakeTiles(Worker, MakeTilesPaths, CtxInterface):
    quality_list: list[str] = None
    decode_check = False

    def init(self):
        self.quality_list = ['0'] + self.ctx.quality_list

    def main(self):
        self.init()
        for _ in self.iterate_name_projection_tiling_tile_quality():
            with task(self):
                self.make_tile()

    def make_tile(self):
        self.assert_tiles()
        self.assert_lossless()

        cmd = self.make_tile_cmd()
        run_command(cmd=cmd,
                    folder=self.tile_folder,
                    log_file=self.tile_log,
                    ui_prefix='\t')

    def assert_tiles(self):
        try:
            self.check_tile_video()
            raise AbortError('')
        except FileNotFoundError:
            self.clean_tile()

    def check_tile_video(self):
        compressed_file_size = self.tile_video.stat().st_size
        compressed_log_text = self.tile_log.read_text()

        if compressed_file_size == 0:
            raise FileNotFoundError('Filesize is 0')

        if 'encoded 1800 frames' not in compressed_log_text:
            self.logger.register_log('Tile log is corrupt', self.tile_log)
            raise FileNotFoundError

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.logger.register_log('Codec version is different.', self.tile_log)
            raise FileNotFoundError

    def assert_lossless(self):
        try:
            self.check_lossless()
        except FileNotFoundError:
            self.logger.register_log('lossless_video not found.', self.lossless_video)
            raise AbortError(f'lossless_video not found.')

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

    def check_lossless(self):
        lossless_video_size = self.lossless_video.stat().st_size
        if lossless_video_size == 0:
            raise FileNotFoundError('lossless_video not found.')

    def clean_tile(self):
        self.tile_log.unlink(missing_ok=True)
        self.tile_video.unlink(missing_ok=True)
