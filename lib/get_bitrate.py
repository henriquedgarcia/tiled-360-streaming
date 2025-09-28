import os
from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.make_dash import MakeDash
from lib.utils.context_utils import task
from lib.utils.io_util import print_error


class GetBitrate(MakeDash):
    cools_names: list
    total_by_name: int

    def init(self):
        self.quality_list.remove('0')
        self.cools_names = ['name', 'projection', 'tiling', 'tile', 'quality',
                            'chunk', 'bitrate']
        self.total_by_name = 181 * len(self.quality_list) * len(self.chunk_list)

    def main(self):
        for _ in self.iterate_name_projection:
            with task(self):
                self.check_bitrate_result_by_name()
                self.get_data()
        self.merge()

    def check_bitrate_result_by_name(self):
        try:
            df = pd.read_pickle(self.bitrate_result_by_name)
        except FileNotFoundError:
            return

        if df.size == self.total_by_name:
            raise AbortError('bitrate_result_by_name is OK.')
        else:
            msg = 'bitrate_result_by_name is NOT OK.'
            self.logger.register_log(msg, self.bitrate_result_by_name)
            raise AbortError(msg)

    def get_data(self):
        data = []
        for n in self.iterate_tiling_tile_quality_chunk:
            print(f'\n\rProcessing {n}/{self.total_by_name} - {self.ctx}', end='')
            bitrate = self.get_bitrate()
            key = (self.name, self.projection, self.tiling,
                   int(self.tile), int(self.quality),
                   int(self.chunk) - 1, bitrate)
            data.append(key)

        print('\nSaving')
        df = pd.DataFrame(data, columns=self.cools_names)
        df.set_index(self.cools_names[:-1], inplace=True)
        pd.to_pickle(df, self.bitrate_result_by_name)

    def get_bitrate(self):
        try:
            bitrate = self.dash_m4s.stat().st_size * 8
        except FileNotFoundError:
            msg = f'dash_m4s not found.'
            self.logger.register_log(msg, self.dash_m4s)
            raise AbortError(msg)

        if bitrate == 0:
            msg = f'dash_m4s is empty.'
            self.logger.register_log(msg, self.dash_m4s)
            raise AbortError(msg)

        return bitrate

    def merge(self):
        merged = None
        for _ in self.iterate_name_projection:
            df = pd.read_pickle(self.bitrate_result_by_name)
            merged = (df if merged is None
                      else pd.concat([merged, df], axis=0))

        if merged.size != len(self.name_list) * self.total_by_name * len(self.projection_list):
            print_error('Dataframe size mismatch.')
            raise AbortError

        merged.to_hdf(self.bitrate_result, key='bitrate_result', mode='w', complevel=9)


if __name__ == '__main__':
    os.chdir('../')

    # config_file = Path('config/config_cmp_qp.json')
    # videos_file = Path('config/videos_reduced.json')

    config_file = Path('config/config_pres_qp.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    app = GetBitrate(ctx)
    app.run()
