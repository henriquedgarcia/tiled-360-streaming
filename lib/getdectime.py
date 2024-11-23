from contextlib import contextmanager
from typing import Any

import numpy as np
from pywin.mfc.object import Object

from lib.assets.autodict import AutoDict
from lib.assets.errors import AbortError
from lib.utils.worker_utils import get_nested_value
from lib.utils.worker_utils import save_json, get_times, print_error
from lib.decode import Decode
from lib.assets.worker import Worker, ProgressBar
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.dectimepaths import DectimePaths
from tqdm import tqdm


class GetDectime(Worker, CtxInterface):
    dectime_result: AutoDict
    dectime_paths: DectimePaths
    t: tqdm

    def iter_proj_tiling_tile_qlt_chunk(self):
        for self.projection in self.projection_list:
            for self.tiling in self.tiling_list:
                for self.tile in self.tile_list:
                    for self.quality in self.quality_list:
                        for self.chunk in self.chunk_list:
                            yield

    total: int

    def init(self):
        self.dectime_paths = DectimePaths(self.ctx)
        self.total = (181 * len(self.projection_list)
                      * len(self.quality_list) * len(self.chunk_list))

    @contextmanager
    def task(self):
        class_name = self.__class__.__name__
        print(f'==== {class_name} {self.ctx} ====')
        self.dectime_result = AutoDict()
        t = ProgressBar(total=self.total, desc=class_name)

        try:
            for _ in self.iter_proj_tiling_tile_qlt_chunk():
                t.update(f'{self.ctx}')
                yield

        except FileNotFoundError as e:
            print_error('Chunk not Found.')
        except AbortError as e:
            print_error(f'\t{e.args[0]}')

        save_json(self.dectime_result, self.dectime_paths.dectime_result_json)
        del t

    def main(self):
        for self.name in self.name_list:
            if self.dectime_paths.dectime_result_json.exists():
                print_error(f'\tThe dectime_result_json exist.')
                continue

            with self.task():
                times = get_times(self.dectime_paths.dectime_log)
                decoded = len(times)

                if decoded < self.config.decoding_num:
                    self.logger.register_log(f'Chunk is not decoded enough. {decoded} times.',
                                             self.dectime_paths.dectime_log)

                self.set_dectime({'dectime': times,
                                  'dectime_avg': np.mean(times),
                                  'dectime_std': np.std(times),
                                  'dectime_med': np.median(times),
                                  })

            save_json(self.dectime_result, self.dectime_paths.dectime_result_json)

    def set_dectime(self, dectime: Any):
        if isinstance(dectime, dict):
            keys = [self.name, self.projection, self.tiling, self.tile, self.quality, self.chunk]
            result = get_nested_value(self.dectime_result, keys)
            result.update(dectime)
        else:
            keys = [self.name, self.projection, self.tiling, self.tile, self.quality]
            result = get_nested_value(self.dectime_result, keys)
            result.update({self.chunk: dectime})
