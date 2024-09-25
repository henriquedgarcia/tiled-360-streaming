import shutil
from lib.assets.status_ctx import StatusCtx
from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.errors import ChunksOkError
from lib.assets.logger import Logger
from lib.utils.context_utils import context_chunk
from lib.utils.worker_utils import decode_video, splitx, print_error, run_command
from lib.assets.worker import Worker
from lib.assets.paths.segmenterpaths import SegmenterPaths


class Others:
    ctx: Context

    def tile_position(self):
        """
        Need video, tiling and tile
        :return: x1, x2, y1, y2
        """
        proj_h, proj_w = self.ctx.video_shape
        tiling_w, tiling_h = splitx(self.ctx.tiling)
        tile_w, tile_h = int(proj_w / tiling_w), int(proj_h / tiling_h)
        tile_m, tile_n = int(self.ctx.tile) % tiling_w, int(self.ctx.tile) // tiling_w
        x1 = tile_m * tile_w
        y1 = tile_n * tile_h
        x2 = tile_m * tile_w + tile_w  # not inclusive [...)
        y2 = tile_n * tile_h + tile_h  # not inclusive [...)
        return x1, y1, x2, y2

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
    #     resolution_ = splitx(self.ctx.scale)
    #     dar = resolution_[0] / resolution_[1]
    #
    #     cmd = (f'bash -c '
    #            f'"'
    #            f'bin/ffmpeg '
    #            f'-hide_banner -y '
    #            f'-ss {self.ctx.offset} '
    #            f'-i {self.config.original_file.as_posix()} '
    #            f'-crf 0 '
    #            f'-t {self.config.duration} '
    #            f'-r {self.config.fps} '
    #            f'-map 0:v '
    #            f'-vf scale={self.ctx.scale},setdar={dar} '
    #            f'{self.segmenter_paths.lossless_video.as_posix()}'
    #            f'"')
    #
    #     print('\t', cmd)
    #
    #     self.segmenter_paths.lossless_video.parent.mkdir(parents=True, exist_ok=True)
    #     process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    #     self.segmenter_paths.lossless_log.write_text(process.stdout)


class CheckTiles:
    logger: Logger
    status: StatusCtx
    segmenter_paths: SegmenterPaths
    config: Config
    ctx: Context

    def check_tile(self, decode_check=False):
        try:
            self.assert_tiles(decode_check=decode_check)
        except FileNotFoundError:
            print(f'\tTiles not Found.')
            print(f'\tChecking lossless')
            self.assert_lossless()
            raise FileNotFoundError

    def assert_tiles(self, decode_check=False):
        print(f'\tChecking tiles.')
        if not self.status.get_status('tile_ok'):
            try:
                self.assert_tile_log()
                self.assert_tile_video()
                self.status.update_status('tile_ok', True)

                if decode_check:
                    self.assert_decode_check()
            except FileNotFoundError as e:
                self.clean_tile()
                self.status.update_status('tile_ok', False)
                self.status.update_status('tile_decode_ok', False)
                raise e

        return 'all ok'

    def assert_tile_log(self):
        try:
            compressed_log_text = self.segmenter_paths.tile_log.read_text()
        except FileNotFoundError:
            raise FileNotFoundError(f'Tile log not exist.')

        if 'encoded 1800 frames' not in compressed_log_text:
            self.logger.register_log('Tile log is corrupt', self.segmenter_paths.tile_log)
            raise FileNotFoundError('Tile log is corrupt')

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.logger.register_log('Codec version is different.', self.segmenter_paths.tile_log)
            raise FileNotFoundError('Codec version is different.')

        return 'all ok'

    def assert_tile_video(self):
        try:
            compressed_file_size = self.segmenter_paths.tile_video.stat().st_size
        except FileNotFoundError:
            raise FileNotFoundError(f'Tile not exist.')

        if compressed_file_size == 0:
            self.logger.register_log('Tile size == 0.', self.segmenter_paths.tile_video)
            raise FileNotFoundError('Tile size == 0.')
        return 'all ok'

    def assert_decode_check(self):
        print(f'\tDecoding tile', end='')

        if self.status.get_status('tile_decode_ok'):
            print_error(f'. OK')
            return 'decode all ok'

        self.assert_one_tile_decode()
        print_error(f'. OK')
        return 'decode all ok'

    def assert_one_tile_decode(self):
        stdout = decode_video(self.segmenter_paths.tile_video)
        if "frame= 1800" not in stdout:
            self.logger.register_log(f'Decoding Compress Error.', self.segmenter_paths.tile_video)
            print_error(f'\tDecode Compressed Video Error. Cleaning.')
            raise FileNotFoundError('Decoding Compress Error')
        return stdout

    def assert_lossless(self):
        if not self.status.get_status('lossless_ok'):
            if not self.segmenter_paths.lossless_video.exists():
                raise AbortError(f'Need create tile but lossless_video not found.')
            self.status.update_status('lossless_ok', True)

    def clean_tile(self):
        self.segmenter_paths.tile_log.unlink(missing_ok=True)
        self.segmenter_paths.tile_video.unlink(missing_ok=True)


def assert_one_chunk_video(ctx, logger, chunk_video):
    try:
        segment_file_size = chunk_video.stat().st_size
    except FileNotFoundError:
        logger.register_log(f'chunk{ctx.chunk} not exist.', chunk_video)
        raise FileNotFoundError(f'video chunk{ctx.chunk} not exist.')

    if segment_file_size == 0:
        logger.register_log(f'Chunk video size == 0', chunk_video)
        raise FileNotFoundError('Chunk video size == 0.')


class CheckChunks:
    logger: Logger
    status: StatusCtx
    segmenter_paths: SegmenterPaths
    config: Config
    ctx: Context

    def check_chunks(self, decode_check=False):
        if not self.status.get_status('segmenter_ok'):
            try:
                self.assert_chunks()
                self.status.update_status('segmenter_ok', True)
                if decode_check:
                    self.assert_chunks_decode()
                    self.status.update_status('chunks_decode_ok', True)
            except FileNotFoundError:
                self.clean_segmenter()
                self.status.update_status('segmenter_ok', False)
                self.status.update_status('chunks_decode_ok', False)
                return

        raise ChunksOkError(f'Segmenter is OK. Skipping.')

    def assert_chunks(self):
        self.assert_segmenter_log()
        self.assert_chunks_video()

    def assert_segmenter_log(self):
        try:
            segment_log_txt = self.segmenter_paths.segmenter_log.read_text()
        except FileNotFoundError:
            raise FileNotFoundError('Segmenter log not exist.')

        with context_chunk(self.ctx, f'{self.config.n_chunks - 1}'):
            segment_video = self.segmenter_paths.chunk_video.as_posix()
        # gambiarra. Todos os logs do teste estão com as pastas antigas.
        segment_video_changed = f'{segment_video}'.replace('chunks', 'segments')

        if f'{segment_video}' not in segment_log_txt and f'{segment_video_changed}' not in segment_log_txt:
            self.logger.register_log('Segmenter log is corrupt.', self.segmenter_paths.segmenter_log)
            raise FileNotFoundError('Segmenter log is corrupt.')

        return 'all ok'

    def assert_chunks_video(self):
        with context_chunk(self.ctx, None):
            for self.ctx.chunk in self.ctx.chunk_list:
                assert_one_chunk_video(self.ctx, self.logger, self.segmenter_paths.chunk_video)
        return 'all ok'

    def assert_chunks_decode(self):
        print(f'\tDecoding chunks', end='')
        with context_chunk(self.ctx, None):
            for self.ctx.chunk in self.ctx.chunk_list:
                print('.', end='')
                self.assert_one_chunk_decode()
            print(f'. OK')

    def assert_one_chunk_decode(self):
        chunk_video = self.segmenter_paths.chunk_video
        stdout = decode_video(chunk_video)
        if "frame=   30" not in stdout:  # specific for ffmpeg 5.0
            self.logger.register_log(f'Segment Decode Error.', chunk_video)
            raise FileNotFoundError(f'Chunk Decode Error.')
        return stdout

    def clean_segmenter(self):
        self.segmenter_paths.segmenter_log.unlink(missing_ok=True)
        shutil.rmtree(self.segmenter_paths.chunks_folder, ignore_errors=True)


class Segmenter(Worker, Others, CheckTiles, CheckChunks):
    def main(self):
        self.segmenter_paths = SegmenterPaths(self.config, self.ctx)
        self.ctx.quality_list = ['0'] + self.ctx.quality_list
        self.create_segments()

    def create_segments(self, decode_check=False):
        for _ in self.iterate_name_projection_quality_tiling_tile():
            print(f'==== Segmenter {self.ctx} ====')
            try:
                self.segmenter(decode_check=decode_check)
            except AbortError as e:
                print_error(f'\t{e.args[0]}')

    def iterate_name_projection_quality_tiling_tile(self):
        for self.ctx.name in self.ctx.name_list:
            for self.ctx.projection in self.ctx.projection_list:
                for self.ctx.quality in self.ctx.quality_list:
                    for self.ctx.tiling in self.ctx.tiling_list:
                        for self.ctx.tile in self.ctx.tile_list:
                            yield

    def segmenter(self, decode_check=False):
        print(f'\tChecking chunks')
        self.check_chunks(decode_check=decode_check)

        print(f'\tChecking tiles')
        try:
            self.check_tile(decode_check=decode_check)
        except FileNotFoundError:
            print(f'\tCreating Tiles ')
            self.make_tiles(decode_check=decode_check)

        print(f'\tChunking Tiles ')
        self.make_chunks()

        print(f'\tChecking chunks again.')
        self.check_chunks(decode_check=decode_check)
        if not self.status.get_status('segmenter_ok'):
            raise AbortError('Error creating chunks. See log.')

    def make_tiles(self, decode_check=False):
        cmd = self.make_compress_tile_cmd()
        print('\t' + cmd)
        run_command(cmd, self.segmenter_paths.tile_folder, self.segmenter_paths.tile_log)

        try:
            self.assert_tiles(decode_check=decode_check)
        except FileNotFoundError:
            raise AbortError(f'Cant create Compressed video.')

    def make_compress_tile_cmd(self):
        lossless_file = self.segmenter_paths.lossless_video.as_posix()
        compressed_file = self.segmenter_paths.tile_video.as_posix()

        x1, y1, x2, y2 = self.tile_position()

        gop_options = f'keyint={self.config.gop}:min-keyint={self.config.gop}:open-gop=0'
        misc_options = f'scenecut=0:info=0'
        qp_options = ':ipratio=1:pbratio=1' if self.config.rate_control == 'qp' else ''
        lossless_option = ':lossless=1' if self.ctx.quality == '0' else ''

        cmd = ('bash -c '
               '"'
               'bin/ffmpeg -hide_banner -y -psnr '
               f'-i {lossless_file} '
               f'-{self.config.rate_control} {self.ctx.quality} -tune psnr '
               f'-c:v libx265 '
               f'-x265-params {gop_options}:{misc_options}{qp_options}{lossless_option} '
               f'-vf crop=w={x2 - x1}:h={y2 - y1}:x={x1}:y={y1} '
               f'{compressed_file}'
               f'"')

        return cmd

    def make_chunks(self):
        cmd = self.make_segmenter_cmd()
        print('\t' + cmd)
        run_command(cmd, self.segmenter_paths.chunks_folder, self.segmenter_paths.segmenter_log)

    def make_segmenter_cmd(self):
        compressed_file = self.segmenter_paths.tile_video.as_posix()
        chunks_folder = self.segmenter_paths.chunks_folder.as_posix()
        cmd = ('bash -c '
               '"'
               f'ffmpeg -hide_banner -i {compressed_file} '
               '-c copy -f segment -segment_time 1 -reset_timestamps 1 '
               f'{chunks_folder}/tile{self.ctx.tile}_%03d.hevc'
               '"')
        return cmd
