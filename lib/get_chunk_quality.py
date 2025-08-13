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
from lib.utils.context_utils import task
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
    total_by_name: int

    def init(self):
        self.cools_names = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk',
                            'frame', 'ssim', 'mse', 's-mse', 'ws-mse']

        self.total_by_name = 181 * len(self.quality_list) * len(self.chunk_list)

    def main(self):
        for _ in self.iterate_name_projection:
            with task(self):
                self.check_chunk_quality_result_by_name()
                self.get_data()
        self.merge()

    def check_chunk_quality_result_by_name(self):
        if self.chunk_quality_result_by_name.exists():
            raise AbortError('chunk_quality_result_by_name is OK.')

        try:
            df = pd.read_pickle(self.chunk_quality_result_by_name)
            if df.size == self.total_by_name * 30 * 4:
                raise AbortError('chunk_quality_result_by_name is OK.')
        except FileNotFoundError:
            pass

    def get_data(self):
        data = []
        for n in self.iterate_tiling_tile_quality_chunk:
            print(f'\rProcessing {n}/{self.total_by_name} - {self.ctx}', end='')

            tile_chunk_quality_dict: dict = load_json(self.chunk_quality_json)
            ssim, mse, s_mse, ws_mse = tile_chunk_quality_dict.values()
            for frame, (ssim_, mse_, s_mse_, ws_mse_) in enumerate(zip(ssim.values(), mse.values(), s_mse.values(), ws_mse.values())):
                key = (self.name, self.projection, self.tiling, int(self.tile), int(self.quality), int(self.chunk) - 1, frame,
                       ssim_, mse_, s_mse_, ws_mse_)
                data.append(key)

        print('\nSaving')
        df = pd.DataFrame(data, columns=self.cools_names)
        df.set_index(self.cools_names[:-4], inplace=True)
        df.to_pickle(self.chunk_quality_result_by_name)

    def merge(self):
        merged = None

        for _ in self.iterate_name_projection:
            df = pd.read_pickle(self.chunk_quality_result_by_name)
            df = df.groupby(['name', 'projection', 'tiling', 'tile', 'quality', 'chunk']).mean()
            merged = (df if merged is None else
                      pd.concat([merged, df], axis=0))

        if merged.size != len(self.name_list)* len(self.projection_list)* 181 * len(self.quality_list) * len(self.chunk_list) * 4:
            print_error('Dataframe size mismatch.')
            raise AbortError

        merged.to_hdf(self.chunk_quality_result, key='chunk_quality_result', mode='w', complevel=9)


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
        axes: np.ndarray

        if quality_plot_file.exists():
            print_error(f'The file quality_result_img exist. Skipping.')
            return

        print(f'\r{self.ctx}', end='')

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
        axes = axes.ravel()
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
        axes = np.ravel(axes)
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

    videos_file = Path('config/videos_reduced.json')
    config_file1 = Path('config/config_cmp_qp.json')

    config = Config(config_file1, videos_file)
    ctx = Context(config=config)

    app = GetChunkQuality(ctx)
    app.run()
