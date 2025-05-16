import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.worker import Worker
from lib.make_chunk_quality import MakeChunkQualityPaths
from lib.utils.util import print_error, load_json


class GetChunkQuality(Worker, MakeChunkQualityPaths):
    """
           The result dict have a following structure:
        results[video_name][tile_pattern][quality][tile_id][chunk_id]
                ['times'|'rate']
        [video_proj]    : The video projection
        [video_name]    : The video name
        [tile_pattern]  : The tile tiling. e.g. "6x4"
        [quality]       : Quality. An int like in crf or qp.
        [tile_id]       : the tile number. ex. max = 6*4
        [chunk_id]      : the chunk number. Start with 1.

        'MSE': float
        'SSIM': float
        'WS-MSE': float
        'S-MSE': float
    """
    data: list
    cools_names: list
    metric_list: list

    def init(self):
        self.cools_names = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk',
                            'frame', 'ssim', 'mse', 's-mse', 'ws-mse']
        self.metric_list = ['ssim', 'mse', 's-mse', 'ws-mse']
        self.metric = 'ssim'

    def __str__(self):
        return f'{self.name}_{self.projection}_{self.tiling}_tile{self.tile}_{self.rate_control}{self.quality}_chunk{self.chunk}'

    def main(self):
        for _ in self.iterate_name_projection:
            if self.chunk_quality_result_by_name.exists(): continue

            try:
                self.get_data()
            except FileNotFoundError:
                print(f'\nFileNotFoundError: {self.chunk_quality_json} not found.')
                self.logger.register_log('FileNotFoundError', self.chunk_quality_json)
                continue

            print('\tSaving Data')
            self.save_data()
            print('\tfinished')

        self.merge()

    def get_data(self):
        self.data = []
        for _ in self.iterate_tiling_tile_quality_chunk:
            print(f'\r{self}', end='')

            tile_chunk_quality_dict = load_json(self.chunk_quality_json)

            metric_values = tuple(tile_chunk_quality_dict.values())
            for frame, values in enumerate(zip(*metric_values)):
                data = (self.name, self.projection, self.tiling,
                        int(self.tile), int(self.quality),
                        int(self.chunk) - 1, frame) + values
                self.data.append(data)

    def save_data(self):
        df = pd.DataFrame(self.data, columns=self.cools_names)
        df.set_index(self.cools_names[:-4], inplace=True)
        df.sort_index(inplace=True)
        for self.metric in self.metric_list:
            new_df = df[[self.metric]]
            new_df.to_pickle(self.chunk_quality_result_by_name)

    def merge(self):
        for self.metric in self.metric_list:
            # if self.chunk_quality_result.exists():
            #     print('chunk_quality_result is OK.')
            #     return
            print(f'Merging {self.metric}...')
            merged = None

            for _ in self.iterate_name_projection:
                df = pd.read_pickle(self.chunk_quality_result_by_name)
                merged = (df if merged is None else
                          pd.concat([merged, df], axis=0))

            if merged.size != 13032000:
                print_error('Dataframe size mismatch.')
                raise AbortError

            merged.to_pickle(self.chunk_quality_result)


class MakePlot(GetChunkQuality):
    _skip: bool
    change_flag: bool
    folder: Path
    results: dict

    def main3(self):
        def get_serie(value1, value2):
            # self.results[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk][self.metric]
            results = self.results[self.projection][self.name][self.tiling][self.quality]

            return [np.average(results[value2][chunk][value1]) for chunk in self.chunk_list]

        for self.video in self.name_list:
            folder = self.quality_folder / '_metric plots' / f'{self.name}'
            folder.mkdir(parents=True, exist_ok=True)
            self.results = load_json(self.chunk_quality_result_by_name)
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    # self.get_tile_image()
                    quality_plot_file = folder / f'{self.tiling}_crf{self.quality}.png'
                    self.make_tile_image(self.metric_list, self.tile_list, quality_plot_file, get_serie, nrows=2,
                                         ncols=2, figsize=(8, 5), dpi=200)

    def main2(self):
        for self.name in self.name_list:
            self.results = load_json(self.chunk_quality_result_by_name)
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        self.get_tile_image()

    def make_tile_image(self, iter1, iter2, quality_plot_file: Path, get_serie: Callable, nrows=1, ncols=1,
                        figsize=(8, 5), dpi=200):
        if quality_plot_file.exists():
            print_error(f'The file quality_result_img exist. Skipping.')
            return

        print(f'\r{self.ctx}', end='')

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
        axes: list[plt.Axes] = list(np.ravel(list(axes)))
        fig: plt.Figure

        for i, value1 in enumerate(iter1):
            for value2 in iter2:
                result = get_serie(value1, value2)
                axes[i].plot(result, label=f'{value2}')
            axes[i].set_title(value1)

        fig.suptitle(f'{self.ctx}')
        fig.tight_layout()
        # fig.show()
        fig.savefig(self.quality_result_img)
        plt.close()

    chunk_results: dict

    def get_tile_image(self):
        if self.quality_result_img.exists():
            print_error(f'The file quality_result_img exist. Skipping.')
            return

        print(f'\rProcessing [{self.name}][{self.projection}][{self.tiling}][crf{self.quality}]', end='')

        fig, axes = plt.subplots(2, 2, figsize=(8, 5), dpi=200)
        axes: list[plt.Axes] = list(np.ravel(axes))
        fig: plt.Figure

        for i, self.metric in enumerate(self.metric_list):
            for self.tile in self.tile_list:
                result = [np.average(self.chunk_results[self.metric]) for self.chunk in self.chunk_list]
                axes[i].plot(result, label=f'{self.tile}')
            axes[i].set_title(self.metric)

        fig.suptitle(f'{self.ctx}')
        fig.tight_layout()
        # fig.show()
        fig.savefig(self.quality_result_img)
        plt.close()


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

    GetChunkQuality(ctx)
