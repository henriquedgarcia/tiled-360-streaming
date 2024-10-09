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
                if not self.tile_is_ok(decode_check):
                    self.create_tile()
                if not self.skip_dash():
                    self.create_dash()
                self.create_mp4(decode_check)

    def create_tile(self):
        print(f'\tCreating Tiles ')
        cmd = self.make_tile_cmd()
        run_command(cmd, self.segmenter_paths.tile_folder, self.segmenter_paths.tile_log, ui_prefix='\t')

    def make_tile_cmd(self):
        lossless_file = self.segmenter_paths.lossless_video.as_posix()
        compressed_file = self.segmenter_paths.tile_video.as_posix()

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

    def clean_tile(self):
        self.segmenter_paths.tile_log.unlink(missing_ok=True)
        self.segmenter_paths.tile_video.unlink(missing_ok=True)

    def tile_is_ok(self, decode_check):
        print(f'\tChecking tiles')
        if not (self.tile_log_is_ok() and self.tile_video_is_ok(decode_check)):
            self.clean_tile()
            return False
        return True

    def lossless_is_ok(self):
        print(f'\tChecking lossless')
        if not self.segmenter_paths.lossless_video.exists():
            self.logger.register_log('lossless_video not found.', self.segmenter_paths.lossless_video)
            raise AbortError(f'lossless_video not found.')

    def tile_video_is_ok(self, decode_check):
        try:
            compressed_file_size = self.segmenter_paths.tile_video.stat().st_size
        except FileNotFoundError:
            return False

        if compressed_file_size == 0:
            return False

        if decode_check:
            print(f'\r\t\tDecoding check tile{self.tile}.', end='')
            self.decode_check()
        return True

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

    def tile_log_is_ok(self):
        try:
            compressed_log_text = self.segmenter_paths.tile_log.read_text()
        except FileNotFoundError:
            return False

        if 'encoded 1800 frames' not in compressed_log_text:
            self.logger.register_log('Tile log is corrupt', self.segmenter_paths.tile_log)
            self.segmenter_paths.tile_log.unlink()
            return False

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.logger.register_log('Codec version is different.', self.segmenter_paths.tile_log)
            self.segmenter_paths.tile_log.unlink(missing_ok=True)
            return False
        return True

    def skip_dash(self):
        print(f'\tChecking dash.')
        try:
            segment_log_txt = self.segmenter_paths.segmenter_log.read_text()
            if f'Dashing P1 AS#1.1(V) done (60 segs)' not in segment_log_txt:
                shutil.rmtree(self.segmenter_paths.mpd_folder, ignore_errors=True)
                self.segmenter_paths.segmenter_log.unlink()
                self.logger.register_log('Segmenter log is corrupt.', self.segmenter_paths.segmenter_log)
                raise FileNotFoundError
            return True
        except FileNotFoundError:
            shutil.rmtree(self.segmenter_paths.mpd_folder, ignore_errors=True)
            self.segmenter_paths.segmenter_log.unlink(missing_ok=True)

        if not self.segmenter_paths.tile_video.exists():
            raise AbortError(f'Tile video not found. Aborting.')
        return False

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

    def create_mp4(self, decode_check):
        print(f'\tCreating mp4 chunks.')
        for self.chunk in self.chunk_list:
            if self.decodable_is_ok(decode_check):
                return
            self.cat_chunk()
        self.chunk = None
        print('\t')

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
