import os
import shutil
from abc import ABC
from pathlib import Path

from py360tools import ERP, CMP, ProjectionBase

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.logger import Logger
from lib.assets.paths.makedashpaths import MakeDashPaths
from lib.assets.paths.maketilespaths import MakeTilesPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import run_command


class MakeTiles(ABC, MakeTilesPaths):
    tile_position: tuple[int, int, int, int]
    proj_obj: dict[str, dict[str, ProjectionBase]]
    logger: Logger

    def make_tile(self):
        self.assert_lossless()

        proj_obj = self.proj_obj[self.projection][self.tiling]
        tile = proj_obj.tile_list[int(self.tile)]
        (y, x), (n, m) = tile.position, tile.shape
        self.tile_position = (y, x, y + n, x + m)

        error = ''
        for self.quality in self.quality_list:
            try:
                self.assert_tiles()
                continue
            except FileNotFoundError:
                pass

            run_command(cmd=self.make_tile_cmd(),
                        folder=self.tile_folder,
                        log_file=self.tile_log,
                        ui_prefix='\t')

            try:
                self.assert_tiles()
            except FileNotFoundError:
                self.logger.register_log('Cant encode the video tile.', self.tile_video)
                error += f'{self.rate_control}{self.quality}'
        if error:
            raise AbortError(f'Cant encode the quality {error} video tile.')
        self.quality = None

    def assert_tiles(self):
        """
        Can raise TilesOkError, FileNotFoundError,
        :return:
        """
        try:
            self.check_tile_video()
        except FileNotFoundError as e:
            self.tile_log.unlink(missing_ok=True)
            self.tile_video.unlink(missing_ok=True)
            raise e

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


class MakeDash(ABC, MakeDashPaths):
    remove = False
    logger: Logger

    def make_dash(self):
        self.assert_dash()

        print(f'\tTile ok. Creating chunks.')
        cmd = self.make_dash_cmd_mp4box()
        run_command(cmd, self.mpd_folder, self.mp4box_log, ui_prefix='\t')

    def assert_dash(self):
        try:
            self._assert_segmenter_log()
            raise AbortError('')
        except FileNotFoundError:
            self._clean_dash()

    def make_dash_cmd_mp4box(self):
        # test gop: "python3 tiled-360-streaming/bin/gop/gop_all.py"
        filename = []
        for self.quality in self.quality_list:
            self.assert_tile_video()
            filename.append(self.tile_video.as_posix())
        self.quality = None

        filename_ = ' '.join(filename)

        cmd = ('bash -c '
               "'"
               'bin/MP4Box '
               '-dash 1000 -frag 1000 -rap '
               '-segment-name %s_ '
               '-profile live '
               f'-out {self.dash_mpd.as_posix()} '
               f'{filename_}'
               "'")
        return cmd

    def assert_tile_video(self):
        try:
            self._check_tile_video()
        except FileNotFoundError:
            self.logger.register_log('Tile video not found.', self.tile_video)
            raise AbortError(f'Tile video not found. Aborting.')

        return False

    def _check_tile_video(self):
        if self.tile_video.stat().st_size == 0:
            self.tile_video.unlink()
            raise FileNotFoundError

    def _assert_segmenter_log(self):
        segment_log_txt = self.mp4box_log.read_text()
        if f'Dashing P1 AS#1.1(V) done (60 segs)' not in segment_log_txt:
            if self.remove:
                self.mp4box_log.unlink(missing_ok=True)
            self.logger.register_log('Segmenter log is corrupt.', self.mp4box_log)
            raise FileNotFoundError

    def _clean_dash(self):
        if self.remove:
            shutil.rmtree(self.mpd_folder, ignore_errors=True)
            self.mp4box_log.unlink(missing_ok=True)


class MakeDecodable(Worker, MakeTiles, MakeDash):
    proj_obj: AutoDict

    def init(self):
        self.proj_obj = AutoDict()
        for self.projection in self.projection_list:
            for self.tiling in self.tiling_list:
                self.proj_obj[f'{self.projection}'][f'{self.tiling}'] = (ERP if self.projection == 'erp' else CMP)(proj_res=self.proj_res, tiling=self.tiling)

    @property
    def iter_name_projection_tiling_tile(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        # cada tile tem suas qualidades e chunks e seu dash
                        yield

    def main(self):
        for _ in self.iter_name_projection_tiling_tile:
            with task(self):
                self.make_tile()
                self.make_dash()


if __name__ == '__main__':
    os.chdir('../')
    # config_file = Path('config/config_cmp_qp.json')
    # videos_file = Path('config/videos_reduced.json')

    config_file = Path('config/config_pres_qp.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)
    MakeDecodable(ctx).run()
