import numpy as np
import pandas as pd

from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.worker import Worker
from lib.utils.util import get_times, print_error


class GetDectime(Worker, DectimePaths):
    def init(self):
        self.projection = 'cmp'
        self.index_names = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk']
        self.cools_names = ['dectime']

    index_names: list
    cools_names: list
    data: list

    def main(self):
        self.data = []
        total = 28 * 181
        n = iter(range(total))
        for self.name in self.name_list:
            for self.tiling in self.tiling_list:
                for self.tile in self.tile_list:
                    print(f'\r{next(n)}/{total} - {self.__class__.__name__} - {self.name}_{self.tiling}_tile{self.tile}                   ', end='')
                    for self.quality in self.quality_list:
                        for self.chunk in self.chunk_list:
                            dectime = self.get_dectime()
                            key = (self.name, self.projection, self.tiling,
                                   int(self.tile), int(self.quality),
                                   int(self.chunk) - 1, dectime)
                            self.data.append(key)

        print('\nSaving Pickle')
        self.save_pickle(self.dectime_result_pickle)
        print('finished')

    def save_pickle(self, filename):
        df = pd.DataFrame(self.data, columns=self.index_names + self.cools_names)
        df.set_index(self.index_names[:-1], inplace=True)
        df.sort_index(inplace=True)
        pd.to_pickle(df, filename)

    def get_dectime(self):
        try:
            times = get_times(self.dectime_log)
        except FileNotFoundError:
            times = []

        if len(times) < self.config.decoding_num:
            msg = f'Chunk is not decoded enough. {len(times)} times.'
            print_error(msg)
            self.logger.register_log(msg, self.dectime_log)
            raise AbortError(msg)
        return np.average(times)
