from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.worker import Worker, ProgressBar
from lib.makequality import ChunkQualityPaths
from lib.utils.worker_utils import save_json, load_json, print_error, get_nested_value


class GetQuality(Worker, CtxInterface):
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
    tile_quality_result: AutoDict
    chunk_quality_paths: ChunkQualityPaths

    def init(self):
        self.chunk_quality_paths = ChunkQualityPaths(self.ctx)

    def iter_proj_tiling_tile_qlt_chunk(self):
        total = (181 * len(self.projection_list)
                 * len(self.quality_list))

        t = ProgressBar(total=total, desc=self.__class__.__name__)

        for self.projection in self.projection_list:
            for self.tiling in self.tiling_list:
                for self.tile in self.tile_list:
                    for self.quality in self.quality_list:
                        t.update(f'{self.ctx}')
                        for self.chunk in self.chunk_list:
                            yield

    @contextmanager
    def task(self):
        print(f'==== {self.__class__.__name__} {self.ctx} ====')
        self.tile_quality_result = AutoDict()

        try:
            yield
        except AbortError as e:
            print_error(f'\t{e.args[0]}')
            return

        for self.metric in ['ssim', 'mse', 's-mse', 'ws-mse']:
            name = self.chunk_quality_paths.chunk_quality_result_json.stem
            new_name = name + f'_{self.metric}'
            new_file = self.chunk_quality_paths.chunk_quality_result_json.with_stem(new_name)
            save_json(self.tile_quality_result[self.metric], new_file)

    def main(self):
        for self.name in self.name_list:
            with self.task():
                if self.chunk_quality_paths.chunk_quality_result_json.exists():
                    raise AbortError('chunk_quality_result_json exist.')

                for _ in self.iter_proj_tiling_tile_qlt_chunk():
                    tile_chunk_quality_dict = self.get_chunk_quality()
                    for self.metric in ['ssim', 'mse', 's-mse', 'ws-mse']:
                        self.set_quality_result(tile_chunk_quality_dict[self.metric])

    def get_chunk_quality(self):
        try:
            tile_chunk_quality_dict = load_json(self.chunk_quality_paths.chunk_quality_json)
        except FileNotFoundError:
            self.logger.register_log('chunk_quality_json not found', self.chunk_quality_paths.chunk_quality_json)
            raise AbortError('tile_chunk_quality_json not found.')
        return tile_chunk_quality_dict

    def set_quality_result(self, chunk_quality_dict):
        keys = [self.metric, self.name, self.projection, self.tiling, self.tile, self.quality, ]
        result = get_nested_value(self.tile_quality_result, keys)
        result.update({self.chunk: chunk_quality_dict})


class MakePlot(GetQuality):
    _skip: bool
    change_flag: bool
    folder: Path
    results: dict

    def main(self):
        def get_serie(value1, value2):
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
