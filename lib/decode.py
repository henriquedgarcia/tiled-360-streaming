from lib.assets.errors import AbortError, DecodeOkError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.worker import Worker
from lib.segmenter import assert_one_chunk_video
from lib.utils.worker_utils import decode_video, print_error, count_decoding


class Decode(Worker):
    turn: int
    segmenter_paths: SegmenterPaths
    dectime_paths: DectimePaths

    def main(self):
        self.segmenter_paths = SegmenterPaths(config=self.config, context=self.ctx)
        self.dectime_paths = DectimePaths(config=self.config, context=self.ctx, segmenter_paths=self.segmenter_paths)
        for self.ctx.attempt in range(self.config.decoding_num):
            for _ in self.iter_decode():
                print(f'==== Decoding {self.ctx} ====')
                try:
                    self.decode()
                except (DecodeOkError, AbortError) as e:
                    print_error(f'\t{e.args[0]}')

    def iter_decode(self):
        for self.ctx.name in self.ctx.name_list:
            for self.ctx.projection in self.ctx.projection_list:
                for self.ctx.quality in self.ctx.quality_list:
                    for self.ctx.tiling in self.ctx.tiling_list:
                        for self.ctx.tile in self.ctx.tile_list:
                            for self.ctx.chunk in self.ctx.chunk_list:
                                yield

    def decode_chunks(self):
        for self.ctx.attempt in range(self.config.decoding_num):
            for _ in self.iter_decode():
                print(f'==== Decoding {self.ctx} ====')
                try:
                    self.decode()
                except (DecodeOkError, AbortError) as e:
                    print_error(f'\t{e.args[0]}')

    def decode(self):
        print(f'\tAttempt {self.ctx.attempt + 1}/{self.config.decoding_num}')
        print(f'\tChecking dectime')
        self.check_dectime()

        print(f'\tChecking chunks')
        self.check_chunk()

        print(f'\tDecoding {int(self.ctx.turn) + 1}/{self.config.decoding_num}')
        self.decode_decode()

    def decode_decode(self):
        dectime_log = self.dectime_paths.dectime_log
        folder = dectime_log.parent
        chunk_video = self.segmenter_paths.chunk_video

        stdout = decode_video(chunk_video, threads=1)

        if not folder.exists():
            folder.mkdir(parents=True)

        with open(dectime_log, 'a') as f:
            f.write('\n' + stdout)

        self.check_dectime()
        raise AbortError(f'Decoded {self.ctx.turn} times.')

    def check_dectime(self):
        try:
            self.assert_dectime()
        except FileNotFoundError:
            self.status.update_status('dectime_ok', False)
            return
        finally:
            self.status.update_status('decode_turn', self.ctx.turn)

        if self.ctx.turn >= self.config.decoding_num:
            self.status.update_status('dectime_ok', True)
            raise DecodeOkError(f'Dectime is OK. Skipping.')

    def assert_dectime(self):
        if not self.status.get_status('dectime_ok'):
            self.assert_dectime_log()

    def assert_dectime_log(self):
        try:
            self.ctx.turn = count_decoding(self.dectime_paths.dectime_log)
        except FileNotFoundError:
            self.ctx.turn = 0
            raise FileNotFoundError('dectime_log not exist.')
        except UnicodeDecodeError:
            self.dectime_paths.dectime_log.unlink()
            self.ctx.turn = 0
            raise FileNotFoundError('dectime_log UnicodeDecodeError.')

    def check_chunk(self):
        try:
            self.assert_chunk()
        except FileNotFoundError:
            print_error(f'\tChunks not Found.')
            self.logger.register_log('\tChunk not exist.', self.segmenter_paths.chunk_video)
            raise AbortError(f'Chunk not exist.')
        self.status.update_status('chunk_ok', True)

    def assert_chunk(self):
        if not self.status.get_status('chunk_ok'):
            assert_one_chunk_video(self.ctx, self.logger,
                                   self.segmenter_paths.chunk_video)

    def clean_dectime_log(self):
        self.dectime_paths.dectime_log.unlink(missing_ok=True)

    def make_decode_cmd(self, threads=1):
        cmd = (f'bin/ffmpeg -hide_banner -benchmark '
               f'-codec hevc '
               f'{"" if not threads else f"-threads {threads} "}'
               f'-i {self.segmenter_paths.chunk_video.as_posix()} '
               f'-f null -')
        cmd = f'bash -c "{cmd}"'

        return cmd
