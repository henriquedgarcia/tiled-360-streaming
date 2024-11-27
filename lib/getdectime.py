from contextlib import contextmanager
from typing import Any

import numpy as np

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.worker import Worker, ProgressBar
from lib.utils.worker_utils import get_nested_value, save_json, get_times, print_error


class GetDectime(Worker, CtxInterface):
    dectime_result: AutoDict
    dectime_paths: DectimePaths

    def iter_proj_tiling_tile_qlt_chunk(self):
        total = (181 * len(self.projection_list)
                 * len(self.quality_list) * len(self.chunk_list))
        t = ProgressBar(total=total, desc=self.__class__.__name__)

        for self.projection in self.projection_list:
            for self.tiling in self.tiling_list:
                for self.tile in self.tile_list:
                    for self.quality in self.quality_list:
                        for self.chunk in self.chunk_list:
                            t.update(f'{self.ctx}')
                            yield

    def init(self):
        self.dectime_paths = DectimePaths(self.ctx)

    @contextmanager
    def task(self):
        print(f'==== {self.__class__.__name__} {self.ctx} ====')
        self.dectime_result = AutoDict()

        try:
            yield
        except AbortError as e:
            print_error(f'\t{e.args[0]}')
            return

        save_json(self.dectime_result, self.dectime_paths.dectime_result_json)

    def main(self):
        for self.name in self.name_list:
            with self.task():
                if self.dectime_paths.dectime_result_json.exists():
                    AbortError(f'The dectime_result_json exist.')

                for _ in self.iter_proj_tiling_tile_qlt_chunk():
                    times = self.get_times()
                    self.set_dectime({'dectime': times,
                                      'dectime_avg': np.mean(times),
                                      'dectime_std': np.std(times),
                                      'dectime_med': np.median(times),
                                      })

    def get_times(self):
        times = get_times(self.dectime_paths.dectime_log)
        decoded = len(times)

        try:
            assert decoded >= self.config.decoding_num
        except AssertionError:
            msg = f'Chunk is not decoded enough. {decoded} times.'
            self.logger.register_log(msg,
                                     self.dectime_paths.dectime_log)
            raise AbortError(f'Chunk is not decoded enough. {decoded} times.')

        return times

    def set_dectime(self, dectime: Any):
        if isinstance(dectime, dict):
            keys = [self.name, self.projection, self.tiling, self.tile, self.quality, self.chunk]
            result = get_nested_value(self.dectime_result, keys)
            result.update(dectime)
        else:
            keys = [self.name, self.projection, self.tiling, self.tile, self.quality]
            result = get_nested_value(self.dectime_result, keys)
            result.update({self.chunk: dectime})
