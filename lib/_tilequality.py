from collections import defaultdict
from pathlib import Path
from time import time
from typing import Union, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse

from ._tiledecodebenchmark import TileDecodeBenchmarkPaths, Utils
from .assets import Log, AutoDict, print_error
from .transform import ea2erp, ea2cmp_face
from .util import save_json, load_json, save_pickle, load_pickle, iter_frame, splitx


class SegmentsQualityPaths(TileDecodeBenchmarkPaths):
    _quality_folder = Path('quality')

    @property
    def quality_folder(self) -> Path:
        folder = self.quality_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def video_quality_folder(self) -> Path:
        folder = self.quality_folder / self.basename2
        # folder = self.project_path / self.quality_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def video_quality_csv(self) -> Path:
        return self.video_quality_folder / f'tile{self.tile}_{int(self.chunk):03d}.csv'

    @property
    def quality_result_img(self) -> Path:
        folder = self.quality_folder / '_metric plots' / f'{self.video}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.tiling}_crf{self.quality}.png'


class SegmentsQualityProps(SegmentsQualityPaths, Utils, Log):
    tile_position: dict
    change_flag: bool
    error: bool
    chunk_quality_df: pd.DataFrame
    sph_points_mask: np.ndarray
    weight_ndarray: Union[np.ndarray, object]
    tile_mask: np.ndarray
    chunk_quality: dict[str, list]
    results: AutoDict
    old_tile: str
    results_dataframe: pd.DataFrame
    method = dict[str, Callable]
    original_quality = '0'
    metric_list = ['MSE', 'SSIM', 'WS-MSE', 'S-MSE']

    def init(self):
        self.sph_points_mask = np.zeros(0)
        self.weight_ndarray = np.zeros(0)
        self.old_tile = ''

    def load_weight_ndarray(self):
        weight_ndarray_file = Path(f'datasets/'
                                   f'weight_ndarray'
                                   f'_{self.proj}'
                                   f'_{self.resolution}'
                                   f'.pickle')

        try:
            self.weight_ndarray = load_pickle(weight_ndarray_file)
            return
        except FileNotFoundError:
            self.make_weight_ndarray()
            save_pickle(self.weight_ndarray, weight_ndarray_file)

    def make_weight_ndarray(self):
        proj_h, proj_w = self.video_shape[:2]

        if self.proj == 'erp':
            pi_proj = np.pi / proj_h
            proj_h_2 = 0.5 - proj_h / 2

            def func(y, x):
                w = np.cos((y + proj_h_2) * pi_proj)
                return w

        elif self.proj == 'cmp':
            r = proj_h / 4
            r1 = 0.5 - r
            r2 = r ** 2

            def func(y, x):
                x = x % r
                y = y % r
                d = (x + r1) ** 2 + (y + r1) ** 2
                w = (1 + d / r2) ** (-1.5)
                return w
        else:
            raise ValueError(f'Wrong self.vid_proj. Value == {self.proj}')

        self.weight_ndarray = np.fromfunction(func, (proj_h, proj_w), dtype='float')

    def load_sph_file(self):
        """
        Load 655362 sample points (elevation, azimuth). Angles in degree.

        :return:
        """

        sph_points_mask_file = Path(f'datasets/'
                                    f'sph_points'
                                    f'_{self.proj}'
                                    f'_{self.resolution}_mask'
                                    f'.pickle')

        try:
            self.sph_points_mask = load_pickle(sph_points_mask_file)
            return
        except FileNotFoundError:
            self.process_sphere_file()
            save_pickle(self.sph_points_mask, sph_points_mask_file)

    def process_sphere_file(self):
        self.sph_points_mask = np.zeros(self.video_shape)
        sph_file = Path('datasets/sphere_655362.txt')
        sph_file_lines = sph_file.read_text().splitlines()[1:]
        # for each line (sample), convert to cartesian system and horizontal system
        for line in sph_file_lines:
            el, az = list(map(np.deg2rad, map(float, line.strip().split())))  # to rad

            if self.proj == 'erp':
                m, n = ea2erp(np.array([[az], [el]]), self.video_shape)
            elif self.proj == 'cmp':
                (m, n), face = ea2cmp_face(np.array([[az], [el]]), self.video_shape)
            else:
                raise ValueError(f'wrong value to {self.proj=}')

            self.sph_points_mask[n, m] = 1

    def read_video_quality_csv(self):
        try:
            self.chunk_quality_df = pd.read_csv(self.video_quality_csv, encoding='utf-8')
        except FileNotFoundError as e:
            print(f'\n\t\tCSV_NOTFOUND_ERROR')
            self.log('CSV_NOTFOUND_ERROR', self.video_quality_csv)
            raise e
        except pd.errors.EmptyDataError as e:
            self.video_quality_csv.unlink(missing_ok=True)
            print(f'\n\t\tCSV_EMPTY_DATA_ERROR')
            self.log('CSV_EMPTY_DATA_ERROR', self.video_quality_csv)
            raise e

        self.check_video_quality_csv()

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

    def check_video_quality_csv(self):
        if len(self.chunk_quality_df['frame']) != int(self.gop):
            self.video_quality_csv.unlink(missing_ok=True)
            print_error(f'\n\t\tMISSING_FRAMES', end='')
            self.log(f'MISSING_FRAMES', self.video_quality_csv)
            raise FileNotFoundError

        if 1 in self.chunk_quality_df['SSIM'].to_list():
            self.log(f'CSV SSIM has 1.', self.segment_file)
            print_error(f'\n\t\tCSV SSIM has 0.', end='')

        if 0 in self.chunk_quality_df['MSE'].to_list():
            self.log('CSV MSE has 0.', self.segment_file)
            print_error(f'\n\t\tCSV MSE has 0.', end='')

        if 0 in self.chunk_quality_df['WS-MSE'].to_list():
            self.log('CSV WS-MSE has 0.', self.segment_file)
            print_error(f'\n\t\tCSV WS-MSE has 0.', end='')

        if 0 in self.chunk_quality_df['S-MSE'].to_list():
            self.log('CSV S-MSE has 0.', self.segment_file)
            print_error(f'\n\t\tCSV S-MSE has 0.', end='')

    @property
    def chunk_results(self):
        results = self.results
        for state in self.state:
            results = results[state]
        return results


class SegmentsQuality(SegmentsQualityProps):

    def main(self):
        for _ in self.iterator():
            self.work()

    def work(self):
        print(f'\r{self.state_str()}: ', end='')

        if self.skip(): return

        chunk_quality = defaultdict(list)
        start = time()
        iter_reference_segment = iter_frame(self.reference_segment)
        iter_segment = iter_frame(self.segment_file)
        zip_frames = zip(iter_reference_segment, iter_segment)

        for frame, (frame1, frame2) in enumerate(zip_frames):
            print(f'\r\t{frame=}', end='')
            chunk_quality['SSIM'].append(self._ssim(frame1, frame2))
            chunk_quality['MSE'].append(self._mse(frame1, frame2))
            chunk_quality['WS-MSE'].append(self._wsmse(frame1, frame2))
            chunk_quality['S-MSE'].append(self._smse_nn(frame1, frame2))

        pd.DataFrame(chunk_quality).to_csv(self.video_quality_csv, encoding='utf-8', index_label='frame')
        print(f"\n\ttime={time() - start}.")

    def skip(self):
        skip = False
        if not self.segment_file.exists():
            self.log('segment_file NOTFOUND', self.segment_file)
            print_error(f'segment_file NOTFOUND')
            skip = True

        if not self.reference_segment.exists():
            self.log('reference_segment NOTFOUND', self.reference_segment)
            print_error(f'reference_segment NOTFOUND')
            skip = True

        try:
            self.read_video_quality_csv()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.error = True
            print('')
            return skip

        return True

    def iterator(self):
        self.init()
        for self.video in self.video_list:
            self.results = AutoDict()
            shape = self.video_shape[:2]
            if self.weight_ndarray.shape != shape:  # suppose that changed projection, the resolution is changed too.
                self.load_weight_ndarray()
            if self.sph_points_mask.shape != shape:  # suppose that change the projection
                self.load_sph_file()

            for self.tiling in self.tiling_list:
                self.update_tile_position()

                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        x1, y1, x2, y2 = self.tile_position[self.tile]
                        self.tile_mask = self.sph_points_mask[y1:y2, x1:x2]

                        for self.chunk in self.chunk_list:
                            yield

    @staticmethod
    def _mse(im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Images must be only one channel (luminance)
        (height, width) = im_ref.shape()
        "float32" = im_ref.dtype()

        :param im_ref:
        :param im_deg:
        :return:
        """
        # im_sqr_err = (im_ref - im_deg) ** 2
        # mse = np.average(im_sqr_err)
        return mse(im_ref, im_deg)

    @staticmethod
    def _ssim(im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Images must be only one channel (luminance)
        (height, width) = im_ref.shape()
        "float32" = im_ref.dtype()

        :param im_ref:
        :param im_deg:
        :return:
        """
        # im_sqr_err = (im_ref - im_deg) ** 2
        # mse = np.average(im_sqr_err)
        return ssim(im_ref, im_deg,
                    data_range=255.0,
                    gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False)

    def _wsmse(self, im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :return:
        """
        x1, y1, x2, y2 = self.tile_position[self.tile]
        weight_tile = self.weight_ndarray[y1:y2, x1:x2]
        wmse = np.sum(weight_tile * (im_ref - im_deg) ** 2) / np.sum(weight_tile)
        return wmse

    def _smse_nn(self, tile_ref: np.ndarray, tile_deg: np.ndarray):
        """
        Calculate of S-PSNR between two images. All arrays must be on the same
        resolution.

        :param tile_ref: The original image
        :param tile_deg: The image degraded
        :return:
        """
        if self.tile != self.old_tile or self.tile == '0':
            x1, y1, x2, y2 = self.tile_position[self.tile]
            self.tile_mask = self.sph_points_mask[y1:y2, x1:x2]
            self.old_tile = self.tile

        tile_ref_m = tile_ref * self.tile_mask
        tile_deg_m = tile_deg * self.tile_mask

        sqr_dif = (tile_ref_m - tile_deg_m) ** 2

        smse_nn = sqr_dif.sum() / 655362
        return smse_nn

    def _collect_ffmpeg_psnr(self) -> dict[str, float]:
        # deprecated
        def get_psnr(line_txt):
            return float(line_txt.strip().split(',')[3].split(':')[1])

        def get_qp(line_txt):
            return float(line_txt.strip().split(',')[2].split(':')[1])

        psnr = None
        compressed_log = self.compressed_file.with_suffix('.log')
        content = compressed_log.read_text(encoding='utf-8')
        content = content.splitlines()

        for line in content:
            if 'Global PSNR' in line:
                psnr = {'psnr': get_psnr(line),
                        'qp_avg': get_qp(line)}
                break
        return psnr


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
            # if list(self.videos_list).index(self.video) < 3: continue
            print(f'\n{self.video}')
            if self.quality_json_exist(): continue

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

    def quality_json_exist(self, check_result=True):
        try:
            self.results = load_json(self.quality_result_json,
                                     AutoDict)
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

        def get_serie(value1, value2):
            # self.results[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk][self.metric]
            results = self.results[self.proj][self.name][self.tiling][self.quality]

            return [np.average(results[value2][chunk][value1])
                    for chunk in self.chunk_list]

        for self.video in self.video_list:
            folder = self.quality_folder / '_metric plots' / f'{self.video}'
            folder.mkdir(parents=True, exist_ok=True)
            self.results = load_json(self.quality_result_json)
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    # self.get_tile_image()
                    quality_plot_file = folder / f'{self.tiling}_crf{self.quality}.png'
                    self.make_tile_image(self.metric_list, self.tile_list,
                                         quality_plot_file,
                                         get_serie,
                                         nrows=2, ncols=2,
                                         figsize=(8, 5), dpi=200)

    def main2(self):
        for self.video in self.video_list:
            self.results = load_json(self.quality_result_json)
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    self.get_tile_image()

    def make_tile_image(self, iter1, iter2, quality_plot_file: Path,
                        get_serie: Callable,
                        nrows=1, ncols=1,
                        figsize=(8, 5), dpi=200):
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


QualityAssessmentOptions = {'0': SegmentsQuality,
                            '1': CollectQuality,
                            '2': MakePlot,
                            }
