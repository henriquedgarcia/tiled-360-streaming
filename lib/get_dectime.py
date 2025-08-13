import os
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import get_times, print_error


class GetDectime(Worker, DectimePaths):
    data: list
    cools_names: list
    total_by_name: int

    def init(self):
        self.cools_names = ['name', 'projection', 'tiling', 'tile', 'quality',
                            'chunk', 'dectime']
        self.total_by_name = 181 * len(self.quality_list) * len(self.chunk_list)

    def main(self):
        for _ in self.iterate_name_projection:
            with task(self):
                self.check_dectime_result_by_name()
                self.get_data()
        self.merge()

    def check_dectime_result_by_name(self):
        if self.dectime_result_by_name.exists():
            raise AbortError('dectime_result_by_name is OK.')

        try:
            df = pd.read_pickle(self.dectime_result_by_name)
            if df.size == self.total_by_name:
                raise AbortError('dectime_result_by_name is OK.')
        except FileNotFoundError:
            pass

    def get_data(self):
        data = []
        for n in self.iterate_tiling_tile_quality_chunk:
            print(f'\rProcessing {n}/{self.total_by_name} - {self.ctx}', end='')
            dectime = self.get_dectime()
            key = (self.name, self.projection, self.tiling,
                   int(self.tile), int(self.quality),
                   int(self.chunk) - 1, dectime)
            data.append(key)

        print('\nSaving')
        df = pd.DataFrame(data, columns=self.cools_names)
        df.set_index(self.cools_names[:-1], inplace=True)
        df.to_pickle(self.dectime_result_by_name)

    def get_dectime(self):
        try:
            times = get_times(self.dectime_log)
        except FileNotFoundError:
            times = []

        if len(times) < self.config.decoding_num:
            msg = f'Chunk is not decoded enough. {len(times)} times.'
            self.logger.register_log(msg, self.dectime_log)
            raise AbortError(msg)
        return times

    def merge(self):
        merged = None
        for _ in self.iterate_name_projection:
            df = pd.read_pickle(self.dectime_result_by_name)
            df['dectime'] = df['dectime'].apply(np.mean)
            merged = (df if merged is None
                      else pd.concat([merged, df], axis=0))

        if merged.size != 434400*2:
            print_error('Dataframe size mismatch.')
            raise AbortError

        merged.to_hdf(self.dectime_result, key='dectime', mode='w', complevel=9)


if __name__ == '__main__':
    os.chdir('../')

    config_file = Path('config/config_cmp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    app = GetDectime(ctx)
    app.run()
