import numpy as np

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import *
from lib.assets.autodict import AutoDict
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.worker import Worker
from lib.segmenter import assert_chunk
from lib.utils.worker_utils import decode_video, print_error, count_decoding, save_json, load_json, get_nested_value, get_times


class Decode(CtxInterface, Worker):
    turn: int
    segmenter_paths: SegmenterPaths
    dectime_paths: DectimePaths
    changes: bool

    def main(self):
        self.segmenter_paths = SegmenterPaths(config=self.config, context=self.ctx)
        self.dectime_paths = DectimePaths(config=self.config, context=self.ctx, segmenter_paths=self.segmenter_paths)
        self.decode_chunks()

    def decode_chunks(self):
        for self.attempt in range(self.config.decoding_num):
            self.changes = False
            for _ in self.iter_decode():
                print(f'==== Decoding {self.ctx} ====')
                try:
                    self.decode()
                except (DecodeOkError, AbortError, InsufficientDecodingError) as e:
                    print_error(f'\t{e.args[0]}')
                except ValueError:
                    self.logger.register_log('Cant decode Chunk.', self.segmenter_paths.chunk_video)
            if not self.changes:
                break

    def iter_decode(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                yield

    def decode(self):
        print(f'\tAttempt {self.attempt + 1}/{self.config.decoding_num}')
        if not self.status.get_status('dectime_ok'):
            print(f'\tChecking dectime')
            self.check_dectime()

        if not self.status.get_status('chunk_ok'):
            print(f'\tChecking chunks')
            self.check_chunk()

        print(f'\tDecoding {int(self.turn) + 1}/{self.config.decoding_num}')
        self.changes = True
        self.decode_decode()
        print_error(f'Decoded {self.turn} times.')

    def check_dectime(self):
        try:
            self.assert_dectime_log()
            self.status.update_status('dectime_ok', True)
            raise DecodeOkError(f'Dectime is OK. Skipping.')
        except InsufficientDecodingError:
            self.status.update_status('decode_turn', self.turn)
            self.status.update_status('dectime_ok', False)
            return

    def assert_dectime_log(self):
        try:
            self.turn = count_decoding(self.dectime_paths.dectime_log)
        except (FileNotFoundError, UnicodeDecodeError):
            self.turn = 0
            self.clean_dectime_log()

        if self.turn >= self.config.decoding_num: return
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

