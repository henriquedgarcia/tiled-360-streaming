import os
from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.make_viewport_quality import ViewportQuality
from lib.utils.util import print_error, load_json


class GetViewportQuality(ViewportQuality):
    data: list
    cools_names: list
    metric_list: list

    def init(self):
        self.cools_names = ['name', 'projection', 'tiling', 'quality', 'user', 'chunk', 'frame',
                            'mse', 'ssim']
        self.metric_list = ['mse', 'ssim']
        self.metric = 'mse'

    def main(self):
        """
        user_viewport_quality é um dicionário de listas.
        As chaves são 'ssim' e 'mse'. As listas contem
        as métricas usando float64 para cada frame do
        chunk (30 frames)
        """
        for _ in self.iterate_name_projection:
            if self.user_viewport_result_by_name.exists(): continue
            try:
                self.get_data()
                self.save_data()
            except FileNotFoundError:
                print(f'\nFileNotFoundError: {self.user_viewport_quality_json} not found.')
                self.logger.register_log('FileNotFoundError', self.user_viewport_quality_json)
                continue

            print('\tfinished')

        for _ in self.iterate_name_projection:
            if not self.user_viewport_result_by_name.exists():
                return
        self.merge()

    def get_data(self):
        self.data = []
        for _ in self.iterate_tiling_quality_user:
            print(f'\r{self}', end='')
            for self.chunk in self.chunk_list:
                # print(f'\r{self.name}_{self.projection}_{self.tiling}_{self.rate_control}{self.quality}_user{self.user}_chunk{self.chunk}', end='')
                user_viewport_quality: dict = load_json(self.user_viewport_quality_json)
                a=list(user_viewport_quality.values())
                for frame, (mse, ssim) in enumerate(zip(*a)):
                    self.data.append((self.name, self.projection, self.tiling, self.quality, self.user, self.chunk, frame, mse, ssim))
            self.chunk = None
        print('\tSaving Data')

    def save_data(self):
        df = pd.DataFrame(self.data, columns=self.cools_names)
        df.set_index(self.cools_names[:-2], inplace=True)
        df.sort_index(inplace=True)
        df.to_pickle(self.user_viewport_result_by_name)

    def merge(self):
        merged = None
        for self.metric in self.metric_list:
            # if self.chunk_quality_result.exists():
            #     print('chunk_quality_result is OK.')
            #     return
            print(f'Merging {self.metric}...')

            for _ in self.iterate_name_projection:
                df = pd.read_pickle(self.user_viewport_result_by_name)
                merged = (df if merged is None else
                          pd.concat([merged, df], axis=0))

            if merged.size != 13032000:
                print_error('Dataframe size mismatch.')
                raise AbortError

            merged.to_pickle(self.chunk_quality_result)


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

    # config_file = Path('config/config_cmp_qp.json')
    config_file = Path('config/config_erp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    GetViewportQuality(ctx)
