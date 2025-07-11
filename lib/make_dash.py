import shutil
from pathlib import Path

from config.config import Config
from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.makedashpaths import MakeDashPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import run_command


class MakeDash(Worker, MakeDashPaths, CtxInterface):
    quality_list: list[str] = None
    decode_check = False

    def init(self):
        self.quality_list = ['0'] + self.ctx.quality_list
        self.remove = self.config.remove

    def main(self):
        self.init()
        for _ in self.iterate_name_projection_tiling_tile():
            with task(self):
                self.work()

    def work(self):
        self.assert_dash()

        print(f'\tTile ok. Creating chunks.')
        cmd = self.make_dash_cmd_mp4box()
        run_command(cmd, self.mpd_folder, self.mp4box_log, ui_prefix='\t')

    def make_dash_cmd_mp4box(self):
        # test gop: "python3 tiled-360-streaming/bin/gop/gop_all.py"
        filename = []
        for self.quality in self.quality_list:
            self.assert_tile_video()
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
        self.quality = None
        return cmd

    def assert_dash(self):
        try:
            self._assert_segmenter_log()
            raise AbortError('')
        except FileNotFoundError:
            self._clean_dash()

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

    def _clean_dash(self):
        if self.remove:
            shutil.rmtree(self.mpd_folder, ignore_errors=True)
            self.mp4box_log.unlink(missing_ok=True)

    # def make_split_cmd_mp4box(self):
    #     compressed_file = self.tile_video.as_posix()
    #     chunks_folder = self.mpd_folder.as_posix()
    #
    #     cmd = ('bash -c '
    #            '"'
    #            'bin/MP4Box '
    #            '-split 1 '
    #            f'{compressed_file} '
    #            f"-out {chunks_folder}/tile{self.tile}_'$'num%03d$.mp4"
    #            '"')
    #     return cmd
    remove = False

    def _assert_segmenter_log(self):
        segment_log_txt = self.mp4box_log.read_text()
        if f'Dashing P1 AS#1.1(V) done (60 segs)' not in segment_log_txt:
            if self.remove:
                self.mp4box_log.unlink(missing_ok=True)
            self.logger.register_log('Segmenter log is corrupt.', self.mp4box_log)
            raise FileNotFoundError

    # def make_segment_cmd_ffmpeg(self):
    #     compressed_file = self.tile_video.as_posix()
    #     chunks_folder = self.mpd_folder.as_posix()
    #     cmd = ('bash -c '
    #            '"'
    #            f'ffmpeg -hide_banner -i {compressed_file} '
    #            '-c copy -f segment -segment_time 1 -reset_timestamps 1 '
    #            f'{chunks_folder}/tile{self.tile}_%03d.hevc'
    #            '"')
    #     return cmd


# def prepare(self):
#     """
#     deprecated
#     :return:
#     """
#     print(f'==== Preparing {self.ctx} ====')
#     if self.lossless_video.exists():
#         print_error(f'\tThe file {self.lossless_video} exist. Skipping.')
#         return
#
#     if not self.original_file.exists():
#         self.logger.register_log(f'The original_file not exist.', self.original_file)
#         print_error(f'\tThe file {self.original_file=} not exist. Skipping.')
#         return
#
#     resolution_ = splitx(self.scale)
#     dar = resolution_[0] / resolution_[1]
#
#     cmd = (f'bash -c '
#            f'"'
#            f'bin/ffmpeg '
#            f'-hide_banner -y '
#            f'-ss {self.offset} '
#            f'-i {self.config.original_file.as_posix()} '
#            f'-crf 0 '
#            f'-t {self.config.duration} '
#            f'-r {self.config.fps} '
#            f'-map 0:v '
#            f'-vf scale={self.scale},setdar={dar} '
#            f'{self.lossless_video.as_posix()}'
#            f'"')
#
#     print('\t', cmd)
#
#     self.lossless_video.parent.mkdir(parents=True, exist_ok=True)
#     process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
#     self.lossless_log.write_text(process.stdout)


if __name__ == '__main__':
    import os

    os.chdir('../')

    # config_file = 'config_erp_qp.json'
    # config_file = 'config_cmp_crf.json'
    # config_file = 'config_erp_crf.json'
    # videos_file = 'videos_reversed.json'
    # videos_file = 'videos_lumine.json'
    # videos_file = 'videos_container0.json'
    # videos_file = 'videos_container1.json'
    # videos_file = 'videos_fortrek.json'
    # videos_file = 'videos_hp_elite.json'
    # videos_file = 'videos_alambique.json'
    # videos_file = 'videos_test.json'
    # videos_file = 'videos_full.json'

    config_file = Path('config/config_cmp_crf.json')
    # config_file = Path('config/config_cmp_qp.json')
    # config_file = Path('config/config_erp_qp.json')
    videos_file = Path('config/videos_reduced.json')
    # videos_file = Path('config/videos_full.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    MakeDash(ctx)
    # CheckViewportQuality(ctx)
