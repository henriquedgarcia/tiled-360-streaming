from pathlib import Path

import numpy as np
from py360tools.transform import ea2nm, ea2nm_face
from py360tools.utils import LazyProperty, splitx
from skimage.metrics import mean_squared_error, structural_similarity

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.utils.worker_utils import load_pickle, save_pickle


class QualityMetrics(CtxInterface):
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

    # segmenter_paths: SegmenterPaths
    #
    # def _collect_ffmpeg_psnr(self) -> dict[str, float]:
    #     # deprecated
    #     def get_psnr(line_txt):
    #         return float(line_txt.strip().split(',')[3].split(':')[1])
    #
    #     def get_qp(line_txt):
    #         return float(line_txt.strip().split(',')[2].split(':')[1])
    #
    #     psnr = None
    #     compressed_log = self.segmenter_paths.segmenter_log
    #     content = compressed_log.read_text(encoding='utf-8')
    #     content = content.splitlines()
    #
    #     for line in content:
    #         if 'Global PSNR' in line:
    #             psnr = {'psnr': get_psnr(line), 'qp_avg': get_qp(line)}
    #             break
    #     return psnr

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
    sph_file = Path('datasets/sphere_655362.txt')
    sph_file_lines = sph_file.read_text().splitlines()[1:]
    ea_array = np.array(list(map(lines_2_list, sph_file_lines))).T

    sph_points_mask = {}
    for proj in ctx.projection_list:
        proj_shape = splitx(ctx.config.config_dict['scale'][proj])[::-1]
        if proj == 'erp':
            nm = ea2nm(ea=ea_array, proj_shape=proj_shape)
        elif proj == 'cmp':
            nm = ea2nm_face(ea=ea_array, proj_shape=proj_shape)[0]
        else:
            raise ValueError(f'Unknown projection: {proj}')
        sph_points_mask[proj] = np.zeros(proj_shape)
        sph_points_mask[proj][nm[0], nm[1]] = 1

    return sph_points_mask


def lines_2_list(line):
    strip_line = line.strip()
    split_line = strip_line.split()
    map_float_line = map(float, split_line)
    map_rad_line = map(np.deg2rad, map_float_line)
    return list(map_rad_line)


def make_weight_ndarray_dict(ctx: Context):
    weight_ndarray_dict_file = Path(f'datasets/weight_ndarray_dict.pickle')

    try:
        weight_ndarray_dict = load_pickle(weight_ndarray_dict_file)
    except FileNotFoundError:
        weight_ndarray_dict = process_weight_ndarray_dict_file(ctx)
        save_pickle(weight_ndarray_dict, weight_ndarray_dict_file)
    return weight_ndarray_dict


def process_weight_ndarray_dict_file(ctx: Context):
    weight_array = {}
    for proj in ctx.projection_list:
        scale = ctx.config.config_dict['scale'][proj]
        w, h = splitx(scale)

        if proj == 'erp':
            pi_proj = np.pi / h
            proj_h_2 = 0.5 - h / 2

            def func(y, x):
                w = np.cos((y + proj_h_2) * pi_proj)
                return w

        elif proj == 'cmp':
            a = h / 2
            r = a / 2
            r1 = 0.5 - r
            r2 = r ** 2

            def func(y, x):
                x = x % a
                y = y % a
                d = (x + r1) ** 2 + (y + r1) ** 2
                w = (1 + d / r2) ** (-1.5)
                return w
        else:
            raise ValueError(f'Unknown projection: {proj}')
        weight_array[proj] = np.fromfunction(func, (h, w), dtype=float)
    # plt.imshow(np.ones((cmp_h, cmp_w)) * 255 * np.fromfunction(func_cmp, (cmp_h, cmp_w), dtype=float));plt.show()
    # plt.imshow(np.ones((erp_h, erp_w)) * 255 * np.fromfunction(func_erp, (erp_h, erp_w), dtype=float));plt.show()
    return weight_array
