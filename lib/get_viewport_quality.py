import os
from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.make_viewport_quality import ViewportQuality
from lib.utils.context_utils import task
from lib.utils.io_util import print_error, load_json


class GetViewportQuality(ViewportQuality):
    """
    user_viewport_quality é um dicionário de listas.
    As chaves são 'ssim' e 'mse'. As listas contem
    as métricas usando float64 para cada frame do
    chunk (30 frames)
    """
    cools_names: list
    total_by_name: int

    def init(self):
        self.cools_names = ['name', 'projection', 'tiling', 'quality', 'user', 'chunk', 'frame',
                            'mse', 'ssim']

    def main(self):
        for _ in self.iterate_name_projection:
            self.total_by_name = len(self.tiling_list) * len(self.quality_list) * len(self.users_list_by_name) * len(self.chunk_list)
            with task(self):
                self.check_user_viewport_result_by_name()
                self.get_data()
        self.merge()

    def check_user_viewport_result_by_name(self):
        try:
            df = pd.read_pickle(self.user_viewport_result_by_name)
            if df.size == self.total_by_name * 30 * 2:
                raise AbortError('user_viewport_result_by_name is OK.')
        except FileNotFoundError:
            pass

    def get_data(self):
        data = []
        for n in self.iterate_tiling_quality_user:
            print(f'\rProcessing {n}/{self.total_by_name} - {self.ctx}', end='')

            user_viewport_quality: dict = load_json(self.user_viewport_quality_json)
            for frame, (mse, ssim) in enumerate(zip(*user_viewport_quality.values())):
                key = (self.name, self.projection, self.tiling, int(self.quality), int(self.user), int(self.chunk) - 1, int(frame), mse, ssim)
                data.append(key)

        print('\nSaving')
        df = pd.DataFrame(data, columns=self.cools_names)
        df.set_index(self.cools_names[:-2], inplace=True)
        df.to_pickle(self.user_viewport_result_by_name)

    def merge(self):
        merged = None
        print(f'Merging...')

        for _ in self.iterate_name_projection:
            df = pd.read_pickle(self.user_viewport_result_by_name)
            df = df.groupby(['name', 'projection', 'tiling', 'quality', 'user', 'chunk']).mean()
            merged = (df if merged is None else
                      pd.concat([merged, df], axis=0))

        if (merged.size != (len(self.name_list)
                            * len(self.projection_list)
                            * len(self.tiling_list)
                            * len(self.quality_list)
                            * len(self.chunk_list) * 30 * 2)):
            print_error('Dataframe size mismatch.')
            raise AbortError

        merged.to_hdf(self.user_viewport_quality_result, key='user_viewport_quality', mode='w', complevel=9)


if __name__ == '__main__':
    os.chdir('../')

    config_file = Path('config/config_cmp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    app = GetViewportQuality(ctx)
    app.run()
