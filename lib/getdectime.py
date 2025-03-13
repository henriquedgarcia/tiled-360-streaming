from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.assets.autodict import AutoDict
from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import get_times, get_nested_value, save_pickle, load_pickle, print_error


class GetDectime(Worker, CtxInterface):
    dectime_result: list
    dectime_paths: DectimePaths
    progress_bar: ProgressBar
    status: AutoDict

    def __init__(self, ctx: Context):
        super().__init__(ctx)
        self.status_file = None

    def iter_name_proj_tiling_tile_qlt_chunk(self):
        for self.name in self.name_list:
            for self.tiling in self.tiling_list:
                for self.tile in self.tile_list:
                    for self.quality in self.quality_list:
                        for self.chunk in self.chunk_list:
                            self.progress_bar.update(f'{self.ctx}')
                            yield

    def init(self):
        self.status_file = Path(f'log/status_{self.__class__.__name__}.pickle')
        self.dectime_paths = DectimePaths(self.ctx)
        if self.dectime_paths.dectime_result_pickle.exists():
            print('file exists')
            exit(0)

        self.projection = 'cmp'
        self.progress_bar = ProgressBar(total=(181
                                               * len(self.quality_list)
                                               * len(self.chunk_list)),
                                        desc=f'{self.__class__.__name__}')

    def main(self):
        dectime_result = []
        for _ in self.iter_name_proj_tiling_tile_qlt_chunk():
            dectime = self.get_dectime()
            dectime_result.append((self.name, self.projection, self.tiling, int(self.tile), int(self.quality), int(self.chunk), dectime))

        result = pd.DataFrame(dectime_result, columns=['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'dectime'])
        result.set_index(['name', 'projection', 'tiling', 'tile', 'quality', 'chunk'], inplace=True)
        save_pickle(result['dectime'], self.dectime_paths.dectime_result_pickle)
        print('finished')

    def get_dectime(self):
        try:
            times = get_times(self.dectime_paths.dectime_log)
        except FileNotFoundError:
            times = []

        if len(times) < self.config.decoding_num:
            msg = f'Chunk is not decoded enough. {len(times)} times.'
            print_error(msg)
            self.logger.register_log(msg, self.dectime_paths.dectime_log)

        return np.average(times)
