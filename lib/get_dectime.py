import os
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.worker import Worker
from lib.utils.util import get_times, print_error


class GetDectime(Worker, DectimePaths):
    data: list
    cools_names: list

    def init(self):
        self.cools_names = ['name', 'projection', 'tiling', 'tile', 'quality',
                            'chunk', 'dectime']

    def main(self):
        for _ in self.iterate_name_projection:
            if self.dectime_result_by_name.exists(): continue

            try:
                self.get_data()
            except FileNotFoundError:
                print(f'\nFileNotFoundError: {self.dectime_log} not found.')
                self.logger.register_log('FileNotFoundError', self.dectime_log)
                continue

            print('\tSaving Data')
            self.save_data()
            print('\tfinished')
        self.merge()

    def merge(self):
        if self.dectime_result.exists():
            print('dectime_result is OK.')
            return

        merged = None

        for _ in self.iterate_name_projection:
            df = pd.read_pickle(self.dectime_result_by_name)
            merged = (df if merged is None
                      else pd.concat([merged, df], axis=0))

        if merged.size != 434400:
            print_error('Dataframe size mismatch.')
            raise AbortError

        merged.to_pickle(self.dectime_result)

    def __str__(self):
        return f'{self.name}_{self.projection}_{self.tiling}_tile{self.tile}_{self.rate_control}{self.quality}_chunk{self.chunk}'

    def get_data(self):
        self.data = []
        for _ in self.iterate_tiling_tile_quality_chunk:
            print(f'\r{self}', end='')

            dectime = self.get_dectime()
            key = (self.name, self.projection, self.tiling,
                   int(self.tile), int(self.quality),
                   int(self.chunk) - 1, dectime)
            self.data.append(key)

    def save_data(self):
        df = pd.DataFrame(self.data, columns=self.cools_names)
        df.set_index(self.cools_names[:-1], inplace=True)
        df.sort_index(inplace=True)
        pd.to_pickle(df, self.dectime_result_by_name)

    def get_dectime(self):
        try:
            times = get_times(self.dectime_log)
        except FileNotFoundError:
            times = []

        if len(times) < self.config.decoding_num:
            msg = f'Chunk is not decoded enough. {len(times)} times.'
            self.logger.register_log(msg, self.dectime_log)
            raise AbortError(msg)
        return np.average(times)


if __name__ == '__main__':
    os.chdir('../')

    # config_file = 'config_erp_qp.json'
    # config_file = 'config_cmp_crf.json'
    # config_file = 'config_erp_crf.json'
    # videos_file = 'videos_reversed.json'
    # videos_file = 'videos_lumine.json'
    # videos_file = 'videos_container0.json'
    # videos_file = 'videos_container1.json'
    # videos_file = 'videos_fortrek.json'
    # videos_file = 'videos_hp_elite.json'
    # videos_file = 'videos_alambique.json'
    # videos_file = 'videos_test.json'
    # videos_file = 'videos_full.json'

    config_file = Path('config/config_cmp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    GetDectime(ctx)
