import numpy as np

from lib.assets.autodict import AutoDict
from lib.assets.errors import AbortError
from lib.utils.worker_utils import get_nested_value
from lib.utils.worker_utils import save_json, get_times, print_error
from lib.decode import Decode
from lib.assets.worker import Worker
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.dectimepaths import DectimePaths
from tqdm import tqdm


class GetDectime(Worker, CtxInterface):
    dectime_result: AutoDict
    dectime_paths: DectimePaths
    t: tqdm

    def iter_decode(self):
        for self.name in self.name_list:
            if self.dectime_paths.dectime_result_json.exists():
                continue

            self.dectime_result = AutoDict()
            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                yield

    def main(self):
        self.dectime_paths = DectimePaths(self.ctx)
        for self.name in self.name_list:
            print(f'==== {self.__class__.__name__} {self.name} ====')

            if self.dectime_paths.dectime_result_json.exists():
                print_error(f'\tThe dectime_result_json exist.')
                continue

            self.dectime_result = AutoDict()

            self.t = tqdm(total=(181
                                 * len(self.projection_list)
                                 * len(self.quality_list)
                                 * len(self.chunk_list)),
                          desc=f'    {self.__class__.__name__}')

            for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
                self.t.set_postfix_str(f'{self.ctx}')
                self.t.update()

                try:
                    times = get_times(self.dectime_paths.dectime_log)
                except FileNotFoundError:
                    raise AbortError('Chunk not Found.')

                decoded = len(times)
                if decoded <= self.config.decoding_num:
                    self.logger.register_log(f'Chunk is not decoded enough. {decoded} times.',
                                             self.dectime_paths.dectime_log)

                self.set_dectime({'dectime': times,
                                  'dectime_avg': np.mean(times),
                                  'dectime_std': np.std(times),
                                  'dectime_med': np.median(times),
                                  })

            save_json(self.dectime_result, self.dectime_paths.dectime_result_json)

    def get_dectime(self):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        return get_nested_value(self.dectime_result, keys)

    def set_dectime(self, value: dict):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        get_nested_value(self.dectime_result, keys).update(value)
