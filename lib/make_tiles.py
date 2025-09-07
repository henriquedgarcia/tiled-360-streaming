import os
from pathlib import Path

from py360tools import ERP, CMP, ProjectionBase

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError, TilesOkError
from lib.assets.paths.maketilespaths import MakeTilesPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import run_command, get_tile_position


class MakeTiles(Worker, MakeTilesPaths):
    tile_position: tuple[int, int, int, int]
    proj_obj: ProjectionBase

    @property
    def iterate_name_projection_tiling_tile_quality(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    self.proj_obj = (ERP if self.projection == 'erp' else CMP)(proj_res=self.proj_res, tiling=self.tiling)
                    for self.tile in self.tile_list:
                        tile = self.proj_obj.tile_list[int(self.tile)]
                        self.tile_position = get_tile_position(tile)
                        for self.quality in self.quality_list:
                            self.ctx.iterations += 1
                            yield

    def main(self):
        for _ in self.iterate_name_projection_tiling_tile_quality:
            with task(self):
                self.make_tile()

    def make_tile(self):
        self.assert_tiles()
        self.assert_lossless()

        cmd = self.make_tile_cmd()
        run_command(cmd=cmd, folder=self.tile_folder, log_file=self.tile_log,
                    ui_prefix='\t')

    def assert_tiles(self):
        try:
            self.check_tile_video()
            raise TilesOkError('')
        except FileNotFoundError:
            self.tile_log.unlink(missing_ok=True)
            self.tile_video.unlink(missing_ok=True)

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
            lossless_video_size = self.lossless_video.stat().st_size
            if lossless_video_size == 0:
                raise FileNotFoundError('lossless_video not found.')

        except FileNotFoundError:
            self.logger.register_log('lossless_video not found.', self.lossless_video)
            raise AbortError(f'lossless_video not found.')

    def make_tile_cmd(self) -> str:
        x1, x2, y1, y2 = self.tile_position
        crop_params = f'crop=w={x2 - x1}:h={y2 - y1}:x={x1}:y={y1}'

        gop_options = f'keyint={self.gop}:min-keyint={self.gop}:open-gop=0'
        misc_options = f':scenecut=0:info=0'
        qp_options = ':ipratio=1:pbratio=1' if self.rate_control == 'qp' else ''
        lossless_option = ':lossless=1' if self.quality == '0' else ''
        codec_params = f'-x265-params {gop_options}{misc_options}{qp_options}{lossless_option}'
        codec = f'-c:v libx265'
        output_options = f'-{self.rate_control} {self.quality} -tune psnr'

        cmd = ('bash -c '
               '"'
               'bin/ffmpeg -hide_banner -y -psnr '
               f'-i {self.lossless_video.as_posix()} '
               f'{output_options} '
               f'{codec} {codec_params} '
               f'-vf {crop_params} '
               f'{self.tile_video.as_posix()}'
               f'"')

        return cmd


if __name__ == '__main__':
    os.chdir('../')
    # config_file = Path('config/config_cmp_qp.json')
    # videos_file = Path('config/videos_reduced.json')

    config_file = Path('config/config_pres_qp.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)
    MakeTiles(ctx).run()
