import os
import shutil
from pathlib import Path

from py360tools import ERP, CMP, Projection, Tile

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError, TilesOkError
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.io_util import print_error
from lib.utils.util import run_command

class MakeChunks(Worker, MakeDecodablePaths):
    tile_position: tuple[int, int, int, int]
    proj_obj: Projection
    remove = False
    tile: Tile
    cmd: str

    def main(self):
        proj_types = {'erp': ERP, 'cmp': CMP}
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                proj = proj_types[self.projection]
                for self.tiling in self.tiling_list:
                    self.proj_obj = proj(proj_res=self.proj_res, tiling=self.tiling)
                    for self.tile in self.proj_obj.tile_list:
                        for self.quality in self.quality_list:
                            with task(self):
                                print_error(' | make_tile ', end='')
                                self.make_tile()

                        with task(self):
                            print_error(' | make_dash ', end='')
                            self.make_dash()

                        for self.quality in self.quality_list:
                            for self.chunk in self.chunk_list:
                                with task(self):
                                    print(' | make_decodable ', end='')
                                    self.make_decodable()

    def make_tile(self):
        self._assert_tiles()
        self._assert_lossless()

        cmd = self._make_tile_cmd()
        run_command(cmd=cmd, folder=self.tile_folder, log_file=self.tile_log,
                    ui_prefix='\t')

    def _assert_tiles(self):
        try:
            self._check_tile_video()
            raise TilesOkError('')
        except FileNotFoundError:
            self.tile_log.unlink(missing_ok=True)
            self.tile_video.unlink(missing_ok=True)

    def _check_tile_video(self):
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

    def _assert_lossless(self):
        try:
            lossless_video_size = self.lossless_video.stat().st_size
            if lossless_video_size == 0:
                raise FileNotFoundError('lossless_video not found.')

        except FileNotFoundError:
            self.logger.register_log('lossless_video not found.', self.lossless_video)
            raise AbortError(f'lossless_video not found.')

    def _make_tile_cmd(self) -> str:
        y1, x1 = self.tile.position
        y2, x2 = self.tile.position + self.tile.shape
        crop_params = f'scale={self.video_shape[1]}:{self.video_shape[0]},crop=w={x2 - x1}:h={y2 - y1}:x={x1}:y={y1}'

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

    def make_dash(self):
        self._assert_dash()

        print(f'\tTile ok. Creating chunks.')
        cmd = self._make_dash_cmd_mp4box()
        run_command(cmd, self.mpd_folder, self.mp4box_log, ui_prefix='\t')

    def _make_dash_cmd_mp4box(self):
        # test gop: "python3 tiled-360-streaming/bin/gop/gop_all.py"
        filename = []
        for self.quality in self.quality_list:
            self._assert_tile_video()
            filename.append(self.tile_video.as_posix())

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

    def _assert_dash(self):
        try:
            self._assert_segmenter_log()
            raise AbortError('')
        except FileNotFoundError:
            self._clean_dash()

    def _assert_segmenter_log(self):
        segment_log_txt = self.mp4box_log.read_text()
        if f'Dashing P1 AS#1.1(V) done (60 segs)' not in segment_log_txt:
            if self.remove:
                self.mp4box_log.unlink(missing_ok=True)
            self.logger.register_log('Segmenter log is corrupt.', self.mp4box_log)
            raise FileNotFoundError

    def _assert_tile_video(self):
        try:
            self._check_tile_video()
        except FileNotFoundError:
            self.logger.register_log('Tile video not found.', self.tile_video)
            raise AbortError(f'Tile video not found. Aborting.')
        return False

    def _clean_dash(self):
        if self.remove:
            shutil.rmtree(self.mpd_folder, ignore_errors=True)
            self.mp4box_log.unlink(missing_ok=True)

    def make_decodable(self):
        self._assert_decodable()
        self._assert_dash2()
        self.make_decodable_cmd()
        self.run_command()

    def _assert_decodable(self):
        try:
            self._check_decodable()
            raise AbortError(f'decodable_chunk is OK.')
        except FileNotFoundError:
            pass

    def _check_decodable(self):
        chunk_size = self.decodable_chunk.stat().st_size
        if chunk_size == 0:
            self.logger.register_log('Chunk size is 0.', self.decodable_chunk)
            self.decodable_chunk.unlink()
            raise FileNotFoundError()

    def _assert_dash2(self):
        msg = []
        if not self.dash_m4s.exists():
            msg += ['dash_m4s not exist']
            self.logger.register_log('Dash M4S not found.', self.dash_m4s)
        if not self.dash_init.exists():
            msg += ['dash_init not exist']
            self.logger.register_log('Dash init not found.', self.dash_m4s)
        if msg:
            raise AbortError('/'.join(msg))

    def make_decodable_cmd(self):
        self.cmd = (f'bash -c "cat {self.dash_init.as_posix()} {self.dash_m4s.as_posix()} '
                    f'> {self.decodable_chunk.as_posix()}"')

    def run_command(self):
        run_command(self.cmd, folder=self.decodable_folder, log_file=None,
                    ui_prefix='\t')

if __name__ == '__main__':
    os.chdir('../')
    # config_file = Path('config/config_cmp_qp.json')
    # videos_file = Path('config/videos_reduced.json')

    config_file = Path('config/config_pres_qp_2.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)
    MakeChunks(ctx).run()
