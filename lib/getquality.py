from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib.assets.autodict import AutoDict
from lib.tilequality import TileQuality
from lib.utils.worker_utils import save_json, load_json, print_error, get_nested_value


class GetQuality(TileQuality):
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
    tile_chunk_quality_dict: AutoDict

    def main(self):
        for self.name in self.name_list:
            print(f'\n{self.name}')
            if self.quality_json_exist(check_result=False): continue

            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                self.work()

            save_json(self.tile_chunk_quality_dict, self.chunk_quality_paths.chunk_quality_result_json)

    error: bool
    change_flag: bool

    def work(self):
        print(f'==== CollectQuality {self.ctx} ====')

        try:
            self.tile_chunk_quality_dict = self.read_video_quality_json()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.error = True
            return

        # https://ffmpeg.org/ffmpeg-filters.html#psnr
        chunk_quality_df = self.tile_chunk_quality_dict[self.metric_list]
        chunk_quality_dict = chunk_quality_df.to_dict(orient='list')

        if self.chunk_results == chunk_quality_dict:
            return
        elif self.change_flag is False:
            print(f'\n\t\tCSV_UPDATED')
            # self.log('CSV_UPDATE', self.video_quality_csv)
            self.change_flag = True

        self.chunk_results.update(chunk_quality_dict)

    def read_video_quality_json(self):
        try:
            tile_chunk_quality_dict = load_json(self.chunk_quality_paths.chunk_quality_json)
        except FileNotFoundError as e:
            self.logger.register_log('CSV_NOTFOUND_ERROR', self.chunk_quality_paths.chunk_quality_json)
            raise FileNotFoundError('tile_chunk_quality_json not found.')
        return tile_chunk_quality_dict

    def quality_json_exist(self, check_result=False):
        try:
            self.tile_chunk_quality_dict = load_json(self.chunk_quality_paths.chunk_quality_result_json, AutoDict)
        except FileNotFoundError:
            self.change_flag = True
            self.tile_chunk_quality_dict = AutoDict()
            return False

        print_error(f'\tThe file quality_result_json exist.')

        if check_result:
            self.change_flag = False
            return False

        return True

    @property
    def chunk_results(self):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        return get_nested_value(self.tile_chunk_quality_dict, keys)

    @chunk_results.setter
    def chunk_results(self, value: dict):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        get_nested_value(self.tile_chunk_quality_dict, keys).update(value)


class MakePlot(GetQuality):
    _skip: bool
    change_flag: bool
    folder: Path
    results: dict

    def main(self):
        def get_serie(value1, value2
                      ):
            # self.results[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk][self.metric]
            results = self.results[self.projection][self.name][self.tiling][self.quality]

            return [np.average(results[value2][chunk][value1]) for chunk in self.chunk_list]

        for self.video in self.name_list:
            folder = self.chunk_quality_paths.base_paths.quality_folder / '_metric plots' / f'{self.name}'
            folder.mkdir(parents=True, exist_ok=True)
            self.results = load_json(self.chunk_quality_paths.chunk_quality_result_json)
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    # self.get_tile_image()
                    quality_plot_file = folder / f'{self.tiling}_crf{self.quality}.png'
                    self.make_tile_image(self.metric_list, self.tile_list, quality_plot_file, get_serie, nrows=2,
                                         ncols=2, figsize=(8, 5), dpi=200)

    def main2(self):
        for self.name in self.name_list:
            self.results = load_json(self.chunk_quality_paths.chunk_quality_result_json)
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
        axes: list[plt.Axes] = list(np.ravel([axes]))
        fig: plt.Figure

        for i, value1 in enumerate(iter1):
            for value2 in iter2:
                result = get_serie(value1, value2)
                axes[i].plot(result, label=f'{value2}')
            axes[i].set_title(value1)

        fig.suptitle(f'{self.ctx}')
        fig.tight_layout()
        # fig.show()
        fig.savefig(self.chunk_quality_paths.quality_result_img)
        plt.close()

    def get_tile_image(self):
        if self.chunk_quality_paths.quality_result_img.exists():
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
        fig.savefig(self.chunk_quality_paths.quality_result_img)
        plt.close()
