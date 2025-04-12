import numpy as np
import pandas as pd

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import get_times, save_pickle, print_error


class GetDectime(Worker, CtxInterface):
    dectime_paths: DectimePaths
    progress_bar: ProgressBar

    def iter_proj_tiling_tile_qlt_chunk(self):
        self.projection = 'cmp'
        self.progress_bar = ProgressBar(total=(len(self.quality_list)
                                               * 181
                                               ),
                                        desc=f'{self.__class__.__name__}')
        for self.tiling in self.tiling_list:
            for self.tile in self.tile_list:
                for self.quality in self.quality_list:
                    self.progress_bar.update(f'{self.ctx}')
                    for self.chunk in self.chunk_list:
                        yield
                    self.chunk = None

    def init(self):
        self.dectime_paths = DectimePaths(self.ctx)
        if self.dectime_paths.dectime_result_pickle.exists():
            print_error('file exists')
            exit(0)

    def main(self):
        for self.name in self.name_list:
            if self.dectime_paths.dectime_result_pickle.exists():
                print_error(f'{self.dectime_paths.dectime_result_pickle} exists')
                continue

            dectime_result = []
            for _ in self.iter_proj_tiling_tile_qlt_chunk():
                dectime = self.get_dectime()
                self.set_dectime(dectime_result, dectime)

            result = pd.DataFrame(dectime_result,
                                  columns=['name', 'projection', 'tiling', 'tile',
                                           'quality', 'chunk', 'dectime'])
            result.set_index(['name', 'projection', 'tiling', 'tile',
                              'quality', 'chunk'], inplace=True)
            save_pickle(result['dectime'], self.dectime_paths.dectime_result_pickle)
            print('finished')

    def set_dectime(self, dectime_result, dectime):
        key = [self.name, self.projection, self.tiling,
               int(self.tile), int(self.quality),
               int(self.chunk) - 1, dectime]
        dectime_result.append(key)

    def get_dectime(self):
        try:
            times = get_times(self.dectime_paths.dectime_log)
        except FileNotFoundError:
            times = []

        if len(times) < self.config.decoding_num:
            msg = f'Chunk is not decoded enough. {len(times)} times.'
            print_error(msg)
            self.logger.register_log(msg, self.dectime_paths.dectime_log)
            raise AbortError(f'{self.dectime_paths.dectime_result_json} not found.')
        return np.average(times)
