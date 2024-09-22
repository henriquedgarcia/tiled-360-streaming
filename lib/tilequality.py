from lib.assets.paths.tilequalitypaths import TileChunkQualityPaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.paths.basepaths import BasePaths
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from py360tools.assets.sph_points import SpherePoints
from py360tools.transform.cmp_transform import ea2nm_face
from py360tools.transform.erp_transform import ea2nm
from py360tools.utils import LazyProperty
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse

from lib.assets.autodict import AutoDict
from lib.assets.context import Context
from lib.assets.worker import Worker
def make_tile_position_dict(video_shape, tiling_list):
    proj_h, proj_w = video_shape
    resolution = f'{video_shape[1]}x{video_shape[0]}'
    tile_position_dict = AutoDict()

    for tiling in tiling_list:
        tiling_m, tiling_n = map(int, splitx(tiling))
        tile_w, tile_h = int(proj_w / tiling_m), int(proj_h / tiling_n)

        for tile in range(tiling_m * tiling_n):
            tile_x = tile % tiling_m
            tile_y = tile // tiling_m
            x1 = tile_x * tile_w  # not inclusive
            x2 = tile_x * tile_w + tile_w  # not inclusive
            y1 = tile_y * tile_h  # not inclusive
            y2 = tile_y * tile_h + tile_h  # not inclusive
            tile_position_dict[resolution][tiling][str(tile)] = [x1, y1, x2, y2]
    return tile_position_dict




from lib.utils.util import save_json, load_json, save_pickle, load_pickle, iter_frame, splitx, print_error
class QualityMetricsProps:
    ctx: Context

    @property
    def tiling(self) -> str:
        return self.ctx.tiling

    @tiling.setter
    def tiling(self, value: str):
        self.ctx.tiling = value

    @property
    def tile(self) -> str:
        return self.ctx.tile

    @tile.setter
    def tile(self, value: str):
        self.ctx.tile = value

    _tile_position_dict: dict = None

    @LazyProperty
    def tile_position_dict(self):
        return make_tile_position_dict(self.ctx.video_shape, self.ctx.tiling_list)


class QualityMetrics(QualityMetricsProps):
    weight_ndarray: np.ndarray

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.mse = mse
        self.ssim = ssim

        self.make_weight_ndarray()
        self.sph_points_mask = self.load_sph_file()

    def make_weight_ndarray(self):
        ...

    def load_sph_file(self) -> np.ndarray:
        """
        Load 655362 sample points (elevation, azimuth). Angles in degree.

        :return:
        """

        sph_points_mask_file = Path(f'datasets/sph_points_mask.pickle')
                                    # f'_{self.ctx.projection}'
                                    # f'_{self.ctx.scale}_mask'

        try:
            sph_points_mask = load_pickle(sph_points_mask_file)
        except FileNotFoundError:
            sph_points_mask = self.process_sphere_file()
            save_pickle(sph_points_mask, sph_points_mask_file)
        return sph_points_mask

    def process_sphere_file(self) -> dict[str, np.ndarray]:
        sph_file = Path('datasets/sphere_655362.txt')
        sph_file_lines = sph_file.read_text().splitlines()[1:]
        sph_points_mask = {}
        for self.ctx.projection in self.ctx.projection_list:
            video_shape = self.ctx.video_shape
            sph_points_mask[self.ctx.projection] = np.zeros(video_shape)

            # for each line (sample), convert to cartesian system and horizontal system
            for line in sph_file_lines:
                el, az = list(map(np.deg2rad, map(float, line.strip().split())))  # to rad

                ea = np.array([[az], [el]])
                proj_shape = video_shape

                if self.ctx.projection == 'erp':
                    m, n = ea2nm(ea=ea, proj_shape=proj_shape)
                elif self.ctx.projection == 'cmp':
                    (m, n), face = ea2nm_face(ea=ea, proj_shape=proj_shape)
                else:
                    raise ValueError(f'Projection must be "erp" or "cmp".')

                sph_points_mask[self.ctx.projection][n, m] = 1
        return sph_points_mask

    def wsmse(self, im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :return:
        """
        x1, y1, x2, y2 = self.tile_position_dict[self.ctx.scale][self.ctx.tiling][self.ctx.tile]
        weight_tile = self.weight_ndarray[self.ctx.scale][self.ctx.tiling][self.ctx.tile][y1:y2, x1:x2]
        wmse = np.sum(weight_tile * (im_ref - im_deg) ** 2) / np.sum(weight_tile)
        return wmse

    tile_mask: np.ndarray

    def smse_nn(self, tile_ref: np.ndarray, tile_deg: np.ndarray):
        """
        Calculate of S-PSNR between two images. All arrays must be on the same
        resolution.

        :param tile_ref: The original image
        :param tile_deg: The image degraded
        :return:
        """
        x1, y1, x2, y2 = self.tile_position_dict[self.tiling][self.tile]
        self.tile_mask = self.sph_points_mask[y1:y2, x1:x2]

        tile_ref_m = tile_ref * self.tile_mask
        tile_deg_m = tile_deg * self.tile_mask

        sqr_dif = (tile_ref_m - tile_deg_m) ** 2

        smse_nn = sqr_dif.sum() / np.sum(self.tile_mask)
        return smse_nn

        # def _collect_ffmpeg_psnr(self) -> dict[str, float]:
        #     # deprecated
        #     def get_psnr(line_txt):
        #         return float(line_txt.strip().split(',')[3].split(':')[1])
        #
        #     def get_qp(line_txt):
        #         return float(line_txt.strip().split(',')[2].split(':')[1])
        #
        #     psnr = None
        #     compressed_log = self.compressed_file.with_suffix('.log')
        #     content = compressed_log.read_text(encoding='utf-8')
        #     content = content.splitlines()
        #
        #     for line in content:
        #         if 'Global PSNR' in line:
        #             psnr = {'psnr': get_psnr(line), 'qp_avg': get_qp(line)}
        #             break
        return psnr


class SegmentsQualityProps:
    ctx: Context

    @property
    def projection(self):
        return self.ctx.projection

    @projection.setter
    def projection(self, value):
        self._projection = value
        pass


def ssim(im_ref: np.ndarray, im_deg: np.ndarray) -> float:
    """
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Images must be only one channel (luminance)
    (height, width) = im_ref.shape()
    "float32" = im_ref.dtype()

    :param im_ref:
    :param im_deg:
    :return:
    """
    return ssim(im_ref, im_deg, data_range=255.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


def mse(im_ref: np.ndarray, im_deg: np.ndarray) -> float:
    """
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Images must be only one channel (luminance)
    (height, width) = im_ref.shape()
    "float32" = im_ref.dtype()

    :param im_ref:
    :param im_deg:
    :return:
    """
    return mse(im_ref, im_deg)


class TileChunkQuality(Worker, SegmentsQualityProps):
    tile_position: dict
    change_flag: bool
    error: bool
    chunk_quality_df: pd.DataFrame
    sph_points_mask: np.ndarray
    tile_mask: np.ndarray
    chunk_quality: dict[str, list]
    results: AutoDict
    old_tile: str
    results_dataframe: pd.DataFrame
    method = dict[str, Callable]
    original_quality = '0'
    metric_list = ['MSE', 'SSIM', 'WS-MSE', 'S-MSE']
    metric_functions: QualityMetrics
    tile_chunk_quality_paths: TileChunkQualityPaths

    def main(self):
        self.tile_chunk_quality_paths = TileChunkQualityPaths(self.config, self.ctx)
        self.tile_chunk_quality_paths = TileChunkQualityPaths(self.config, self.ctx)
        self.init()
        for _ in self.iterator():
            self.work()

    def init(self):
        self.old_tile = ''
        self.ctx.quality_list.remove('0')
        self.ctx.tiling_list.remove('1x1')
        self.metric_functions = QualityMetrics(self.ctx)

    sphere_points: SpherePoints

    def work(self):
        print(f'\r{self.state_str()}: ')

        if self.skip(): return
        chunk_quality = defaultdict(list)
        start = time()
        iter_reference_segment = iter_frame(self.reference_segment)
        iter_segment = iter_frame(self.segment_video)
        zip_frames = zip(iter_reference_segment, iter_segment)

        for frame, (frame1, frame2) in enumerate(zip_frames):
            print(f'\r\t{frame=}', end='')
            chunk_quality['SSIM'].append(self._ssim(frame1, frame2))
            chunk_quality['MSE'].append(self._mse(frame1, frame2))
            chunk_quality['WS-MSE'].append(self._wsmse(frame1, frame2))
            chunk_quality['S-MSE'].append(self._smse_nn(frame1, frame2))
        pd.DataFrame(chunk_quality).to_csv(self.video_quality_csv, encoding='utf-8', index_label='frame')
        print(f"\ttime={time() - start}.")

    def read_video_quality_csv(self):
        try:
            self.chunk_quality_df = pd.read_csv(self.video_quality_csv,
                                                encoding='utf-8',
                                                index_col=0)
        except FileNotFoundError as e:
            print(f'\n\t\tCSV_NOTFOUND_ERROR')
            self.log('CSV_NOTFOUND_ERROR',
                     self.video_quality_csv)
            raise e
        except pd.errors.EmptyDataError as e:
            self.video_quality_csv.unlink(missing_ok=True)
            print(f'\n\t\tCSV_EMPTY_DATA_ERROR')
            self.log('CSV_EMPTY_DATA_ERROR',
                     self.video_quality_csv)
            raise e

        self.check_video_quality_csv()

    def iterator(self):
        for self.name in self.ctx.name_list:
            for self.proj in self.ctx.projection_list:
                for self.tiling in self.ctx.tiling_list:
                    for self.quality in self.ctx.quality_list:
                        for self.tile in self.ctx.tile_list:
                            for self.chunk in self.ctx.chunk_list:
                                self.results = AutoDict()
                                yield

    def check_video_quality_csv(self):
        if len(self.chunk_quality_df['MSE']) != int(self.gop):
            self.video_quality_csv.unlink(missing_ok=True)
            print_error(f'\n\t\tMISSING_FRAMES', end='')
            self.logger(f'MISSING_FRAMES', self.video_quality_csv)
            return False

        if 1 in self.chunk_quality_df['SSIM'].to_list():
            self.logger(f'CSV SSIM has 1.', self.segment_video)
            print_error(f'\n\t\tCSV SSIM has 0.', end='')

        if 0 in self.chunk_quality_df['MSE'].to_list():
            self.logger('CSV MSE has 0.', self.segment_video)
            print_error(f'\n\t\tCSV MSE has 0.', end='')

        if 0 in self.chunk_quality_df['WS-MSE'].to_list():
            self.logger('CSV WS-MSE has 0.', self.segment_video)
            print_error(f'\n\t\tCSV WS-MSE has 0.', end='')

        if 0 in self.chunk_quality_df['S-MSE'].to_list():
            self.logger('CSV S-MSE has 0.', self.segment_video)
            print_error(f'\n\t\tCSV S-MSE has 0.', end='')
        return True

    def update_tile_position(self):
        self.tile_position = {}
        for self.tile in self.tile_list:
            proj_h, proj_w = self.video_shape[:2]

            tiling_m, tiling_n = splitx(self.tiling)
            tile_w, tile_h = int(proj_w / tiling_m), int(proj_h / tiling_n)
            tile_x, tile_y = int(self.tile) % tiling_m, int(self.tile) // tiling_m
            x1, x2 = tile_x * tile_w, tile_x * tile_w + tile_w  # not inclusive
            y1, y2 = tile_y * tile_h, tile_y * tile_h + tile_h  # not inclusive
            self.tile_position[self.tile] = [x1, y1, x2, y2]

    @property
    def chunk_results(self):
        results = self.results
        results = results[self.proj][self.name][self.tiling]
        results = results[self.quality][self.tile][self.chunk]
        return results

    def skip(self):
        skip = False
        if not self.segment_video.exists():
            self.logger('segment_file NOTFOUND', self.segment_video)
            print_error(f'segment_file NOTFOUND')
            skip = True

        if not self.reference_segment.exists():
            self.logger('reference_segment NOTFOUND', self.reference_segment)
            print_error(f'reference_segment NOTFOUND')
            skip = True

        try:
            self.chunk_quality_df = pd.read_csv(self.video_quality_csv, encoding='utf-8', index_col=0)
        except FileNotFoundError:
            return skip
        except pd.errors.EmptyDataError:
            self.video_quality_csv.unlink(missing_ok=True)
            self.logger('CSV_EMPTY_DATA_ERROR', self.video_quality_csv)
            return skip

        return skip or self.check_video_quality_csv()


class CollectQuality(SegmentsQualityProps):
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

    def main(self):
        # self.get_tile_image()

        for self.video in self.video_list:
            print(f'\n{self.video}')
            if self.quality_json_exist(check_result=False): continue

            self.error = False

            for _ in self.main_loop():
                self.work()

            if self.change_flag and not self.error:
                print('\n\tSaving.')
                save_json(self.results, self.quality_result_json)

    def main_loop(self):
        for self.tiling in self.tiling_list:
            for self.quality in self.quality_list:
                for self.tile in self.tile_list:
                    for self.chunk in self.chunk_list:
                        yield

    def quality_json_exist(self, check_result=False
                           ):
        try:
            self.results = load_json(self.quality_result_json, AutoDict)
        except FileNotFoundError:
            self.change_flag = True
            self.results = AutoDict()
            return False

        print_error(f'\tThe file quality_result_json exist.')

        if check_result:
            self.change_flag = False
            return False

        return True

    def check_qlt_results(self):
        for self.metric in self.metric_list:
            if len(self.chunk_results[self.metric]) != 30:
                break
        else:
            return

    def work(self):
        print(f'\r\t{self.state_str()} ', end='')
        try:
            self.check_qlt_results()
        except KeyError:
            pass

        try:
            self.read_video_quality_csv()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.error = True
            return

        # https://ffmpeg.org/ffmpeg-filters.html#psnr
        chunk_quality_df = self.chunk_quality_df[self.metric_list]
        chunk_quality_dict = chunk_quality_df.to_dict(orient='list')

        if self.chunk_results == chunk_quality_dict:
            return
        elif self.change_flag is False:
            print(f'\n\t\tCSV_UPDATED')
            # self.log('CSV_UPDATE', self.video_quality_csv)
            self.change_flag = True

        self.chunk_results.update(chunk_quality_dict)


class MakePlot(SegmentsQualityProps):
    chunk_quality_df: pd.DataFrame
    _skip: bool
    change_flag: bool
    folder: Path

    def main(self):
        def get_serie(value1, value2
                      ):
            # self.results[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk][self.metric]
            results = self.results[self.proj][self.name][self.tiling][self.quality]

            return [np.average(results[value2][chunk][value1]) for chunk in self.chunk_list]

        for self.video in self.video_list:
            folder = self.quality_folder / '_metric plots' / f'{self.video}'
            folder.mkdir(parents=True, exist_ok=True)
            self.results = load_json(self.quality_result_json)
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    # self.get_tile_image()
                    quality_plot_file = folder / f'{self.tiling}_crf{self.quality}.png'
                    self.make_tile_image(self.metric_list, self.tile_list, quality_plot_file, get_serie, nrows=2,
                                         ncols=2, figsize=(8, 5), dpi=200)

    def main2(self):
        for self.video in self.video_list:
            self.results = load_json(self.quality_result_json)
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    self.get_tile_image()

    def make_tile_image(self, iter1, iter2, quality_plot_file: Path, get_serie: Callable, nrows=1, ncols=1,
                        figsize=(8, 5), dpi=200
                        ):
        if quality_plot_file.exists():
            print_error(f'The file quality_result_img exist. Skipping.')
            return

        print(f'\r{self.state_str()}', end='')

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
        axes: list[plt.Axes] = list(np.ravel([axes]))
        fig: plt.Figure

        for i, value1 in enumerate(iter1):
            for value2 in iter2:
                result = get_serie(value1, value2)
                axes[i].plot(result, label=f'{value2}')
            axes[i].set_title(value1)

        fig.suptitle(f'{self.state_str()}')
        fig.tight_layout()
        # fig.show()
        fig.savefig(self.quality_result_img)
        plt.close()

    def get_tile_image(self):
        if self.quality_result_img.exists():
            print_error(f'The file quality_result_img exist. Skipping.')
            return

        print(f'\rProcessing [{self.proj}][{self.video}][{self.tiling}][crf{self.quality}]', end='')

        fig, axes = plt.subplots(2, 2, figsize=(8, 5), dpi=200)
        axes: list[plt.Axes] = list(np.ravel(axes))
        fig: plt.Figure

        for i, self.metric in enumerate(self.metric_list):
            for self.tile in self.tile_list:
                result = [np.average(self.chunk_results[self.metric]) for self.chunk in self.chunk_list]
                axes[i].plot(result, label=f'{self.tile}')
            axes[i].set_title(self.metric)

        fig.suptitle(f'{self.state_str()}')
        fig.tight_layout()
        # fig.show()
        fig.savefig(self.quality_result_img)
        plt.close()


QualityAssessmentOptions = {'0': TileChunkQuality, '1': CollectQuality, '2': MakePlot, }
