import os
from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.make_dash import MakeDash
from lib.utils.context_utils import task
from lib.utils.util import print_error


class GetBitrate(MakeDash):
    """
    self.video_bitrate[name][projection][tiling][tile]  |
                                  ['dash_mpd'] |
                                  ['dash_init'][self.quality] |
                                  ['dash_m4s'][self.quality][self.chunk]
    :return:
    """
    cools_names: list
    total_by_name: int

    def init(self):
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
            if df.size == self.total_by_name:
                raise AbortError('bitrate_result_by_name is OK.')
        except FileNotFoundError:
            pass

    def get_data(self):
        data = []
        for n in self.iterate_tiling_tile_quality_chunk:
            print(f'\n\rProcessing {n}/{self.total_by_name} - {self.ctx}', end='')
            try:
                bitrate = self.dash_m4s.stat().st_size * 8
            except FileNotFoundError:
                self.logger.register_log('dash_m4s is MISSING', self.dash_m4s)
                raise AbortError('\ndash_m4s is MISSING.')

            key = (self.name, self.projection, self.tiling,
                   int(self.tile), int(self.quality),
                   int(self.chunk) - 1, bitrate)
            data.append(key)

        print('\nSaving')
        cools_names = ('name', 'projection', 'tiling', 'tile', 'quality',
                       'chunk', 'bitrate')
        df = pd.DataFrame(data, columns=cools_names)
        df.set_index(list(cools_names[:-1]), inplace=True)
        pd.to_pickle(df, self.bitrate_result_by_name)

    def merge(self):
        merged = None
        for _ in self.iterate_name_projection:
            df = pd.read_pickle(self.bitrate_result_by_name)
            merged = (df if merged is None
                      else pd.concat([merged, df], axis=0))

        if merged.size != len(self.name_list) * self.total_by_name * 2:
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
