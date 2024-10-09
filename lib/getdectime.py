import numpy as np

from lib.assets.autodict import AutoDict
from lib.utils.worker_utils import get_nested_value
from lib.utils.worker_utils import save_json, get_times, print_error
from lib.decode import Decode


class GetDectime(Decode):
    def iter_decode(self):
        for self.name in self.name_list:
            if self.dectime_paths.dectime_result_json.exists():
                continue

            self.dectime = AutoDict()
            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                yield
            save_json(self.dectime, self.dectime_paths.dectime_result_json)

    def decode_chunks(self):
        for _ in self.iter_decode():
            print(f'==== Getting Dectime {self.ctx} ====')

            try:
                times = get_times(self.dectime_paths.dectime_log)
            except FileNotFoundError:
                print_error('\tChunk not Found.')
                continue

            decoded = len(times)
            if decoded <= self.config.decoding_num:
                self.logger.register_log(f'Chunk is not decoded enough. {decoded} times.',
                                         self.dectime_paths.dectime_log)

            self.set_dectime({'dectime': times,
                              'dectime_avg': np.mean(times),
                              'dectime_std': np.std(times),
                              'dectime_med': np.median(times),
                              })

    dectime: AutoDict

    def get_dectime(self):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        return get_nested_value(self.dectime, keys)

    def set_dectime(self, value: dict):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        get_nested_value(self.dectime, keys).update(value)
