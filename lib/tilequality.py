from collections import defaultdict
from pathlib import Path
from time import time
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from py360tools.transform import ea2nm, ea2nm_face
from py360tools.utils import LazyProperty, splitx
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity

from lib.assets.autodict import AutoDict
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.paths.tilequalitypaths import TileChunkQualityPaths
from lib.assets.worker import Worker
from lib.utils.worker_utils import (make_tile_position_dict, save_json, load_json, iter_frame, print_error, load_pickle,
                                    save_pickle, get_nested_value)


class TileChunkQualityProps:
    ctx: Context

    @property
    def name_list(self):
        return self.ctx.name_list

    @property
    def name(self):
        return self.ctx.name

    @name.setter
    def name(self, value):
        self.ctx.name = value

    @property
    def projection_list(self):
        return self.ctx.projection_list

    @property
    def projection(self):
        return self.ctx.projection

    @projection.setter
    def projection(self, value):
        self.ctx.projection = value

    @property
    def quality_list(self):
        return self.ctx.quality_list

    @property
    def quality(self):
        return self.ctx.quality

    @quality.setter
    def quality(self, value):
        self.ctx.quality = value

    @property
    def tiling_list(self):
        return self.ctx.tiling_list

    @property
    def tiling(self) -> str:
        return self.ctx.tiling

    @tiling.setter
    def tiling(self, value: str):
        self.ctx.tiling = value

    @property
    def tile_list(self):
        return self.ctx.tile_list

    @property
    def tile(self) -> str:
        return self.ctx.tile

    @tile.setter
    def tile(self, value: str):
        self.ctx.tile = value

    @property
    def chunk_list(self):
        return self.ctx.chunk_list

    @property
    def chunk(self):
        return self.ctx.chunk

    @chunk.setter
    def chunk(self, value):
        self.ctx.chunk = value

    @property
    def metric(self):
        return self.ctx.metric

    @metric.setter
    def metric(self, value):
        self.ctx.metric = value

    @property
    def metric_list(self):
        return ["ssim", "mse", "s-mse", "ws-mse"]

    @property
    def video_shape(self):
        return self.ctx.video_shape

    @property
    def scale(self):
        return self.ctx.scale

    @LazyProperty
    def tile_position_dict(self) -> dict:
        """
        tile_position_dict[resolution: str][tiling: str][tile: str]
        :return:
        """
        print(f'\r\tMaking tile position dict')
        return make_tile_position_dict(self.video_shape, self.tiling_list)

    @LazyProperty
    def sph_points_mask_dict(self):
        """
        sph_points_mask[projection: str]
        :return:
        """
        print(f'\r\tMaking sphere points mask')
        return make_sph_points_mask_dict(self.ctx)

    @LazyProperty
    def weight_ndarray(self):
        """
        weight_ndarray[projection: str]
        :return:
        """
        print(f'\r\tMaking weight array')
        return make_weight_ndarray_dict(self.ctx)


class QualityMetrics(TileChunkQualityProps):
    def __init__(self, ctx: Context):
        self.ctx = ctx

    @staticmethod
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
        return mean_squared_error(im_ref, im_deg)

    @staticmethod
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
        return structural_similarity(im_ref, im_deg, data_range=255.0, gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False)

    def wsmse(self, im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :return:
        """
        x1, y1, x2, y2 = self.tile_position_dict[self.scale][self.tiling][self.tile]
        weight_tile = self.weight_ndarray[self.projection][y1:y2, x1:x2]
        wmse = np.sum(weight_tile * (im_ref - im_deg) ** 2) / np.sum(weight_tile)
        return wmse

    # from PIL import Image
    # Image.fromarray(array).show()
    def smse_nn(self, tile_ref: np.ndarray, tile_deg: np.ndarray):
        """
        Calculate of S-PSNR between two images. All arrays must be on the same
        resolution.

        :param tile_ref: The original image
        :param tile_deg: The image degraded
        :return:
        """
        x1, y1, x2, y2 = self.tile_position_dict[self.scale][self.tiling][self.tile]
        tile_mask = self.sph_points_mask_dict[self.projection][y1:y2, x1:x2]

        tile_ref_m = tile_ref * tile_mask
        tile_deg_m = tile_deg * tile_mask

        sqr_dif = (tile_ref_m - tile_deg_m) ** 2

        smse_nn = sqr_dif.sum() / np.sum(tile_mask)
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
        # return psnr


class TileChunkQualityOriginal(Worker, TileChunkQualityProps):
    # metric_list = ['mse', 'SSIM', 'WS-MSE', 'S-MSE']
    quality_metrics: QualityMetrics
    tile_chunk_quality_paths: TileChunkQualityPaths
    segmenter_paths: SegmenterPaths

    def main(self):
        self.init()

        for _ in self.iterator():
            try:
                print(f'==== Segmenter {self.ctx} ====')
                self.work()
            except (AbortError, FileNotFoundError) as e:
                print_error('\t' + e.args[0])

    def init(self):
        self.tile_chunk_quality_paths = TileChunkQualityPaths(self.config, self.ctx)
        self.segmenter_paths = SegmenterPaths(self.config, self.ctx)
        self.quality_metrics = QualityMetrics(self.ctx)

    def iterator(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        if self.tiling =='1x1': continue
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                yield

    def work(self):
        self.check_tile_chunk_quality()
        self.assert_segments()

        start = time()

        chunk_quality = defaultdict(list)
        iter_reference_segment = iter_frame(self.tile_chunk_quality_paths.reference_chunk)
        iter_segment = iter_frame(self.segmenter_paths.chunk_video)
        zip_frames = zip(iter_reference_segment, iter_segment)

        for frame, (frame1, frame2) in enumerate(zip_frames):
            print(f'\r\t{frame=}', end='')
            chunk_quality['ssim'].append(self.quality_metrics.ssim(frame1, frame2))
            chunk_quality['mse'].append(self.quality_metrics.mse(frame1, frame2))
            chunk_quality['s-mse'].append(self.quality_metrics.smse_nn(frame1, frame2))
            chunk_quality['ws-mse'].append(self.quality_metrics.wsmse(frame1, frame2))

        save_json(chunk_quality, self.tile_chunk_quality_paths.tile_chunk_quality_json)
        print(f"\ttime={time() - start}.")

    def check_tile_chunk_quality(self):
        if not self.status.get_status('tile_chunk_quality_json_ok'):
            try:
                self.assert_tile_chunk_quality_json()
            except FileNotFoundError:
                self.status.update_status('tile_chunk_quality_json_ok', False)
                return
            self.status.update_status('tile_chunk_quality_json_ok', True)
        raise AbortError('tile_chunk_quality_json is OK.')

    def assert_tile_chunk_quality_json(self):
        chunk_quality = load_json(self.tile_chunk_quality_paths.tile_chunk_quality_json)

        if len(chunk_quality['MSE']) != int(self.config.gop):
            self.tile_chunk_quality_paths.tile_chunk_quality_json.unlink(missing_ok=True)
            self.logger.register_log(f'MISSING_FRAMES', self.tile_chunk_quality_paths.tile_chunk_quality_json)
            raise FileNotFoundError('Missing Frames')

        msg = []
        if 1 in chunk_quality['SSIM'].to_list():
            self.logger.register_log(f'SSIM has 1.', self.segmenter_paths.chunk_video)
            msg.append('SSIM has 1.')

        if 0 in chunk_quality['MSE'].to_list():
            self.logger.register_log('MSE has 0.', self.segmenter_paths.chunk_video)
            msg.append('MSE has 0.')

        if 0 in chunk_quality['WS-MSE'].to_list():
            self.logger.register_log('WS-MSE has 0.', self.segmenter_paths.chunk_video)
            msg.append('WS-MSE has 0.')

        if 0 in chunk_quality['S-MSE'].to_list():
            self.logger.register_log('S-MSE has 0.', self.segmenter_paths.chunk_video)
            msg.append('S-MSE has 0.')

        if len(msg) != 0:
            msg = "\n\t".join(msg)
            print_error(f'\t{msg}')

    def assert_segments(self):
        self.check_segment_file()
        self.check_reference_chunk()

    def check_segment_file(self):
        if not self.segmenter_paths.chunk_video.exists():
            self.logger.register_log('segment_file NOTFOUND', self.segmenter_paths.chunk_video)
            raise FileNotFoundError('segment_file NOTFOUND')

    def check_reference_chunk(self):
        if not self.tile_chunk_quality_paths.reference_chunk.exists():
            self.logger.register_log('reference_segment NOTFOUND', self.tile_chunk_quality_paths.reference_chunk)
            raise FileNotFoundError('reference_segment NOTFOUND')


class CollectQuality(TileChunkQualityOriginal):
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

            save_json(self.tile_chunk_quality_dict, self.tile_chunk_quality_paths.video_quality_json)

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
            tile_chunk_quality_dict = load_json(self.tile_chunk_quality_paths.tile_chunk_quality_json)
        except FileNotFoundError as e:
            self.logger.register_log('CSV_NOTFOUND_ERROR', self.tile_chunk_quality_paths.tile_chunk_quality_json)
            raise FileNotFoundError('tile_chunk_quality_json not found.')
        return tile_chunk_quality_dict

    def quality_json_exist(self, check_result=False):
        try:
            self.tile_chunk_quality_dict = load_json(self.tile_chunk_quality_paths.video_quality_json, AutoDict)
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


class MakePlot(CollectQuality):
    _skip: bool
    change_flag: bool
    folder: Path

    def main(self):
        def get_serie(value1, value2
                      ):
            # self.results[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk][self.metric]
            results = self.results[self.projection][self.name][self.tiling][self.quality]

            return [np.average(results[value2][chunk][value1]) for chunk in self.chunk_list]

        for self.video in self.name_list:
            folder = self.tile_chunk_quality_paths.base_paths.quality_folder / '_metric plots' / f'{self.name}'
            folder.mkdir(parents=True, exist_ok=True)
            self.results = load_json(self.tile_chunk_quality_paths.video_quality_json)
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    # self.get_tile_image()
                    quality_plot_file = folder / f'{self.tiling}_crf{self.quality}.png'
                    self.make_tile_image(self.metric_list, self.tile_list, quality_plot_file, get_serie, nrows=2,
                                         ncols=2, figsize=(8, 5), dpi=200)

    def main2(self):
        for self.name in self.name_list:
            self.results = load_json(self.tile_chunk_quality_paths.video_quality_json)
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        self.get_tile_image()

    def make_tile_image(self, iter1, iter2, quality_plot_file: Path, get_serie: Callable, nrows=1, ncols=1,
                        figsize=(8, 5), dpi=200
                        ):
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
        fig.savefig(self.tile_chunk_quality_paths.quality_result_img)
        plt.close()

    def get_tile_image(self):
        if self.tile_chunk_quality_paths.quality_result_img.exists():
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
        fig.savefig(self.tile_chunk_quality_paths.quality_result_img)
        plt.close()


def make_sph_points_mask_dict(ctx: Context) -> np.ndarray:
    """
    Load 655362 sample points (elevation, azimuth). Angles in degree.

    :return:
    """

    sph_points_mask_file = Path(f'datasets/sph_points_mask.pickle')

    try:
        sph_points_mask = load_pickle(sph_points_mask_file)
    except FileNotFoundError:
        sph_points_mask = process_sphere_file(ctx)
        save_pickle(sph_points_mask, sph_points_mask_file)
    return sph_points_mask


def process_sphere_file(ctx: Context) -> dict[str, np.ndarray]:
    erp_shape = splitx(ctx.config.config_dict['scale']['erp'])[::-1]
    cmp_shape = splitx(ctx.config.config_dict['scale']['cmp'])[::-1]

    sph_file = Path('datasets/sphere_655362.txt')
    sph_file_lines = sph_file.read_text().splitlines()[1:]
    ea_array = np.array(list(map(lines_2_list, sph_file_lines))).T
    erp_nm = ea2nm(ea=ea_array, proj_shape=erp_shape)
    cmp_nm = ea2nm_face(ea=ea_array, proj_shape=cmp_shape)[0]

    sph_points_mask = {'erp': np.zeros(erp_shape),
                       'cmp': np.zeros(cmp_shape)}
    sph_points_mask['erp'][erp_nm[0], erp_nm[1]] = 1
    sph_points_mask['cmp'][cmp_nm[0], cmp_nm[1]] = 1

    return sph_points_mask


def lines_2_list(line):
    strip_line = line.strip()
    split_line = strip_line.split()
    map_float_line = map(float, split_line)
    map_rad_line = map(np.deg2rad, map_float_line)
    return list(map_rad_line)


def load_weight_ndarray(ctx: Context) -> np.ndarray:
    """
    Load 655362 sample points (elevation, azimuth). Angles in degree.

    :return:
    """

    sph_points_mask_file = Path(f'datasets/weight_ndarray.pickle')

    try:
        sph_points_mask = load_pickle(sph_points_mask_file)
    except FileNotFoundError:
        sph_points_mask = process_sphere_file(ctx)
        save_pickle(sph_points_mask, sph_points_mask_file)
    return sph_points_mask


def make_weight_ndarray_dict(ctx: Context):
    weight_ndarray_dict_file = Path(f'datasets/weight_ndarray_dict.pickle')

    try:
        weight_ndarray_dict = load_pickle(weight_ndarray_dict_file)
    except FileNotFoundError:
        weight_ndarray_dict = process_weight_ndarray_dict_file(ctx)
        save_pickle(weight_ndarray_dict, weight_ndarray_dict_file)
    return weight_ndarray_dict


def process_weight_ndarray_dict_file(ctx: Context):
    erp_scale = ctx.config.config_dict['scale']['erp']
    erp_w, erp_h = splitx(erp_scale)

    pi_proj = np.pi / erp_h
    proj_h_2 = 0.5 - erp_h / 2

    cmp_scale = ctx.config.config_dict['scale']['cmp']
    cmp_w, cmp_h = splitx(cmp_scale)
    a = cmp_h / 2
    r = a / 2
    r1 = 0.5 - r
    r2 = r ** 2

    def func_erp(y, x):
        w = np.cos((y + proj_h_2) * pi_proj)
        return w

    def func_cmp(y, x):
        x = x % a
        y = y % a
        d = (x + r1) ** 2 + (y + r1) ** 2
        w = (1 + d / r2) ** (-1.5)
        return w

    weight_array = {'erp': np.fromfunction(func_erp, (erp_h, erp_w), dtype=float),
                    'cmp': np.fromfunction(func_cmp, (cmp_h, cmp_w), dtype=float)}
    # plt.imshow(np.ones((cmp_h, cmp_w)) * 255 * np.fromfunction(func_cmp, (cmp_h, cmp_w), dtype=float));plt.show()
    # plt.imshow(np.ones((erp_h, erp_w)) * 255 * np.fromfunction(func_erp, (erp_h, erp_w), dtype=float));plt.show()
    return weight_array


TileChunkQuality = TileChunkQualityOriginal
