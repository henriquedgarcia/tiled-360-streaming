import shutil
from contextlib import contextmanager

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.worker import Worker
from lib.utils.worker_utils import decode_video, print_error, run_command


# def prepare(self):
#     """
#     deprecated
#     :return:
#     """
#     print(f'==== Preparing {self.ctx} ====')
#     if self.segmenter_paths.lossless_video.exists():
#         print_error(f'\tThe file {self.segmenter_paths.lossless_video} exist. Skipping.')
#         return
#
#     if not self.segmenter_paths.original_file.exists():
#         self.logger.register_log(f'The original_file not exist.', self.segmenter_paths.original_file)
#         print_error(f'\tThe file {self.segmenter_paths.original_file=} not exist. Skipping.')
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
#            f'{self.segmenter_paths.lossless_video.as_posix()}'
#            f'"')
#
#     print('\t', cmd)
#
#     self.segmenter_paths.lossless_video.parent.mkdir(parents=True, exist_ok=True)
#     process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
#     self.segmenter_paths.lossless_log.write_text(process.stdout)


@contextmanager
def task(self):
    print(f'==== {self.__class__.__name__} {self.ctx} ====')
    try:
        yield
    except AbortError as e:
        print_error(f'\t{e.args[0]}')
    finally:
        pass


class Segmenter(Worker, CtxInterface):
    quality_list: list[str] = None
    segmenter_paths: SegmenterPaths

    def iterate_name_projection_quality_tiling_tile(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.tile in self.tile_list:
                            self.ctx.iterations += 1
                            yield

    def init(self):
        self.segmenter_paths = SegmenterPaths(self.ctx)
        self.quality_list = ['0'] + self.ctx.quality_list

    def main(self):
        self.init()
        decode_check = False
        for _ in self.iterate_name_projection_quality_tiling_tile():
            with task(self):
                if not self.skip_dash():
                    self.create_dash()
                self.create_mp4(decode_check)

    def skip_dash(self):
        print(f'\tChecking dash.')
        try:
            segment_log_txt = self.segmenter_paths.segmenter_log.read_text()
            if f'Dashing P1 AS#1.1(V) done (60 segs)' not in segment_log_txt:
                self.logger.register_log('Segmenter log is corrupt.', self.segmenter_paths.segmenter_log)
                raise FileExistsError
            return True
        except FileNotFoundError:
            shutil.rmtree(self.segmenter_paths.mpd_folder, ignore_errors=True)
            self.segmenter_paths.segmenter_log.unlink(missing_ok=True)

        print(f'\tChunks not found. Checking tile.')
        if not self.segmenter_paths.tile_video.exists():
            raise AbortError(f'Tile video not found. Aborting.')

    def create_dash(self):
        print(f'\tTile ok. Creating chunks.')
        cmd = self.make_segmenter_cmd_mp4box2()
        run_command(cmd, self.segmenter_paths.mpd_folder, self.segmenter_paths.segmenter_log, ui_prefix='\t')

    def make_segmenter_cmd_mp4box_1(self):
        compressed_file = self.segmenter_paths.tile_video.as_posix()
        chunks_folder = self.segmenter_paths.mpd_folder.as_posix()

        cmd = ('bash -c '
               '"'
               'bin/MP4Box '
               '-split 1 '
               f'{compressed_file} '
               f"-out {chunks_folder}/tile{self.tile}_'$'num%03d$.mp4"
               '"')
        return cmd

    def make_segmenter_cmd_mp4box2(self):
        compressed_file = self.segmenter_paths.tile_video.as_posix()
        dash_mpd = self.segmenter_paths.dash_mpd.as_posix()
        'python3 /mnt/d/Henrique/Desktop/tiled-360-streaming/bin/gop/gop_all.py'
        cmd = ('bash -c '
               "'"
               'bin/MP4Box '
               '-dash 1000 -frag 1000 -rap '
               '-segment-name %s_ '
               '-profile live '
               f'-out {dash_mpd} '
               f'{compressed_file}'
               "'")
        return cmd

    def decodable_is_ok(self, decode_check):
        try:
            chunk_size = self.segmenter_paths.decodable_chunk.stat().st_size
            if decode_check:
                print(f'\r\t\tDecoding check chunk{self.chunk}.', end='')
                self.check_one_chunk_decode()
        except FileNotFoundError:
            shutil.rmtree(self.segmenter_paths.decodable_folder, ignore_errors=True)
            return False

        if chunk_size == 0:
            self.logger.register_log('Chunk size is 0.', self.segmenter_paths.decodable_chunk)
            self.segmenter_paths.decodable_chunk.unlink()
            return False

        return True

    def check_one_chunk_decode(self):
        chunk_video = self.segmenter_paths.decodable_chunk
        stdout = decode_video(chunk_video, ui_prefix='', ui_suffix='')
        if "frame=   30" not in stdout:  # specific for ffmpeg 5.0
            self.logger.register_log(f'Chunk Decode Error.', chunk_video)
            raise FileNotFoundError(f'Chunk Decode Error.')

    def create_mp4(self,decode_check):
        print(f'\tCreating mp4 chunks.')
        for self.chunk in self.chunk_list:
            if self.decodable_is_ok(decode_check):
                return
            self.cat_chunk()
        self.chunk = None
        print('')

    def make_segmenter_cmd(self):
        compressed_file = self.segmenter_paths.tile_video.as_posix()
        chunks_folder = self.segmenter_paths.mpd_folder.as_posix()
        cmd = ('bash -c '
               '"'
               f'ffmpeg -hide_banner -i {compressed_file} '
               '-c copy -f segment -segment_time 1 -reset_timestamps 1 '
               f'{chunks_folder}/tile{self.tile}_%03d.hevc'
               '"')
        return cmd

    def cat_chunk(self):
        dash_init = self.segmenter_paths.dash_init.as_posix()
        dash_m4s = self.segmenter_paths.dash_m4s.as_posix()
        dash_chunk_cat = self.segmenter_paths.decodable_chunk.as_posix()
        cmd = (f'bash -c "cat {dash_init} {dash_m4s} '
               f'> {dash_chunk_cat}"')
        run_command(cmd, folder=self.segmenter_paths.decodable_folder, ui_prefix='\t', ui_suffix='.')
