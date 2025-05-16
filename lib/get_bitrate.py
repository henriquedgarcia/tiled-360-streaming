import os
from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.worker import Worker
from lib.make_dash import MakeDashPaths
from lib.utils.util import print_error


class GetBitrate(Worker, MakeDashPaths):
    """
    self.video_bitrate[name][projection][tiling][tile]  |
                                  ['dash_mpd'] |
                                  ['dash_init'][self.quality] |
                                  ['dash_m4s'][self.quality][self.chunk]
    :return:
    """
    data: list
    cools_names: list

    def init(self):
        self.cools_names = ['name', 'projection', 'tiling', 'tile', 'quality',
                            'chunk', 'bitrate']

    def main(self):
        for _ in self.iterate_name_projection:
            if self.bitrate_result_by_name.exists(): continue

            try:
                self.get_data()
            except FileNotFoundError:
                print(f'\nFileNotFoundError: {self.dash_m4s} not found.')
                self.logger.register_log('FileNotFoundError', self.dash_m4s)
                continue

            print('\tSaving Data')
            self.save_data()
            print('\tfinished')

        self.merge()

    def __str__(self):
        return f'{self.name}_{self.projection}_{self.tiling}_{self.tile}_{self.quality}'

    def get_data(self):
        self.data = []
        for _ in self.iterate_tiling_tile_quality_chunk:
            if self.quality == self.quality_list[0]: print(str(self))
            bitrate = self.dash_m4s.stat().st_size * 8

            key = (self.name, self.projection, self.tiling,
                   int(self.tile), int(self.quality),
                   int(self.chunk) - 1, bitrate)
            self.data.append(key)

    def merge(self):
        if self.bitrate_result.exists():
            print('bitrate_result is OK.')
            return

        merged = None

        for _ in self.iterate_name_projection:
            df = pd.read_pickle(self.bitrate_result_by_name)
            merged = (df if merged is None
                      else pd.concat([merged, df], axis=0))

        if merged.size != 434400:
            print_error('Dataframe size mismatch.')
            raise AbortError

        merged.to_pickle(self.bitrate_result)

    def save_data(self):
        df = pd.DataFrame(self.data, columns=self.cools_names)
        df.set_index(self.cools_names[:-1], inplace=True)
        df.sort_index(inplace=True)
        pd.to_pickle(df, self.bitrate_result_by_name)


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

    GetBitrate(ctx)
