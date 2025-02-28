import shutil

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.makedashpaths import MakeDashPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import run_command


class MakeDash(Worker, CtxInterface):
    quality_list: list[str] = None
    make_decodable_path: MakeDashPaths
    decode_check = False

    def init(self):
        self.make_decodable_path = MakeDashPaths(self.ctx)
        self.quality_list = ['0'] + self.ctx.quality_list

    def main(self):
        self.init()
        for _ in self.iterate_name_projection_tiling_tile_quality():
            with task(self):
                self.work()

    def work(self):
        self.assert_dash()

        print(f'\tTile ok. Creating chunks.')
        cmd = self.make_dash_cmd_mp4box()
        run_command(cmd, self.mpd_folder, self.segmenter_log, ui_prefix='\t')

    def assert_dash(self):
        # raise AbortError('Dash is ok. Skipping.')
        try:
            self.assert_segmenter_log()
            raise AbortError('')
        except FileNotFoundError:
            self.clean_dash()

        try:
            self.assert_tile_video()
        except FileNotFoundError:
            self.logger.register_log('Tile video not found.', self.tile_video)
            raise AbortError(f'Tile video not found. Aborting.')

        return False

    def clean_dash(self):
        shutil.rmtree(self.mpd_folder, ignore_errors=True)
        self.segmenter_log.unlink(missing_ok=True)

    def assert_tile_video(self):
        if self.tile_video.stat().st_size == 0:
            self.tile_video.unlink()
            raise FileNotFoundError

    def assert_segmenter_log(self):
        segment_log_txt = self.segmenter_log.read_text()
        if f'Dashing P1 AS#1.1(V) done (60 segs)' not in segment_log_txt:
            self.segmenter_log.unlink(missing_ok=True)
            self.logger.register_log('Segmenter log is corrupt.', self.segmenter_log)
            raise FileNotFoundError

    def make_split_cmd_mp4box(self):
        compressed_file = self.tile_video.as_posix()
        chunks_folder = self.mpd_folder.as_posix()

        cmd = ('bash -c '
               '"'
               'bin/MP4Box '
               '-split 1 '
               f'{compressed_file} '
               f"-out {chunks_folder}/tile{self.tile}_'$'num%03d$.mp4"
               '"')
        return cmd

    def make_dash_cmd_mp4box(self):
        # test gop
        # python3 /mnt/d/Henrique/Desktop/tiled-360-streaming/bin/gop/gop_all.py
        filename = []
        for self.quality in self.quality_list:
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

    def make_segment_cmd_ffmpeg(self):
        compressed_file = self.tile_video.as_posix()
        chunks_folder = self.mpd_folder.as_posix()
        cmd = ('bash -c '
               '"'
               f'ffmpeg -hide_banner -i {compressed_file} '
               '-c copy -f segment -segment_time 1 -reset_timestamps 1 '
               f'{chunks_folder}/tile{self.tile}_%03d.hevc'
               '"')
        return cmd

    @property
    def tile_video(self):
        return self.make_decodable_path.tile_video

    @property
    def mpd_folder(self):
        return self.make_decodable_path.mpd_folder

    @property
    def dash_mpd(self):
        return self.make_decodable_path.dash_mpd

    @property
    def segmenter_log(self):
        return self.make_decodable_path.mp4box_log

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
