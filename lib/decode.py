from lib.assets.errors import AbortError, DecodeOkError, InsufficientDecodingError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.worker import Worker
from lib.segmenter import assert_chunk
from lib.utils.worker_utils import decode_video, print_error, get_times


def get_turn(dectime_log):
    try:
        turn = len(get_times(dectime_log, only_count=True))
    except FileNotFoundError:
        print('ERROR: FileNotFoundError. Return 0.')
        turn = 0
    return turn


class Decode(Worker):
    turn: int
    segmenter_paths: SegmenterPaths
    dectime_paths: DectimePaths
    changes: bool

    def main(self):
        self.segmenter_paths = SegmenterPaths(config=self.config, context=self.ctx)
        self.dectime_paths = DectimePaths(config=self.config, context=self.ctx, segmenter_paths=self.segmenter_paths)
        self.decode_chunks()

    def decode_chunks(self):
        for self.ctx.attempt in range(self.config.decoding_num):
            self.changes = False
            for _ in self.iter_decode():
                print(f'==== Decoding {self.ctx} ====')
                try:
                    self.decode()
                except (DecodeOkError, AbortError) as e:
                    print_error(f'\t{e.args[0]}')
            if not self.changes:
                break

    def iter_decode(self):
        for self.ctx.name in self.ctx.name_list:
            for self.ctx.projection in self.ctx.projection_list:
                for self.ctx.quality in self.ctx.quality_list:
                    for self.ctx.tiling in self.ctx.tiling_list:
                        for self.ctx.tile in self.ctx.tile_list:
                            for self.ctx.chunk in self.ctx.chunk_list:
                                yield

    def decode(self):
        print(f'\tAttempt {self.ctx.attempt + 1}/{self.config.decoding_num}')
        if not self.status.get_status('dectime_ok'):
            print(f'\tChecking dectime')
            self.check_dectime()

        if not self.status.get_status('chunk_ok'):
            print(f'\tChecking chunks')
            self.check_chunk()

        print(f'\tDecoding {int(self.ctx.turn) + 1}/{self.config.decoding_num}')
        self.decode_decode()
        self.changes = True

    def check_dectime(self):
        try:
            self.assert_dectime_log()
            self.status.update_status('dectime_ok', True)
            raise DecodeOkError(f'Dectime is OK. Skipping.')
        except InsufficientDecodingError:
            self.status.update_status('decode_turn', self.ctx.turn)
            self.status.update_status('dectime_ok', False)
            return

    def assert_dectime_log(self):
        try:
            self.ctx.turn = get_turn(self.dectime_paths.dectime_log)
        except (FileNotFoundError, UnicodeDecodeError):
            self.ctx.turn = 0
            self.clean_dectime_log()

        if self.ctx.turn >= self.config.decoding_num: return
        raise InsufficientDecodingError(f'Dectime is OK. Skipping.')

    def clean_dectime_log(self):
        self.dectime_paths.dectime_log.unlink(missing_ok=True)

    def check_chunk(self):
        try:
            assert_chunk(self.ctx, self.logger, self.segmenter_paths.chunk_video)
            self.status.update_status('chunk_ok', True)
        except FileNotFoundError:
            print_error(f'\tChunks not Found.')
            self.status.update_status('chunk_ok', False)
            self.logger.register_log('\tChunk not exist.', self.segmenter_paths.chunk_video)
            raise AbortError(f'Chunk not exist.')

    def decode_decode(self):
        dectime_log = self.dectime_paths.dectime_log

        stdout = decode_video(self.segmenter_paths.chunk_video, threads=1)

        if not dectime_log.parent.exists():
            dectime_log.parent.mkdir(parents=True)

        with open(dectime_log, 'a') as f:
            f.write('\n' + stdout)

        self.check_dectime()
        raise AbortError(f'Decoded {self.ctx.turn} times.')
