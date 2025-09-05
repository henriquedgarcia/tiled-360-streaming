from pathlib import Path

import numpy as np
from py360tools import ERP, CMP
from py360tools.utils import splitx
from skimage.metrics import mean_squared_error, structural_similarity

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.utils.util import save_pickle, load_pickle


class QualityMetrics(CtxInterface):
    ctx: Context

    def __init__(self, make_chunk_quality_obj: CtxInterface):
        self.ctx = make_chunk_quality_obj.ctx
        self.tile_position = self.ctx.proj_obj.tile_list[int(self.tile)].position
        self.sph_points_mask_dict = self.make_sph_points_mask_dict()
        self.weight_ndarray = make_weight_ndarray_dict(self.ctx)

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
        Must be the same size
        :param im_ref:
        :param im_deg:
        :return:
        """
        x1, x2, y1, y2 = self.ctx.tile_position
        weight_tile = self.weight_ndarray[self.projection][y1:y2, x1:x2]

        wmse = np.sum(weight_tile * (im_ref - im_deg) ** 2) / np.sum(weight_tile)
        return wmse

    # from PIL import Image
    # Image.fromarray(array).show()
    def smse_nn(self, tile_ref: np.ndarray, tile_deg: np.ndarray) -> float:
        """
        Calculate of S-PSNR between two images. All arrays must be on the same
        resolution.

        :param tile_ref: The original image
        :param tile_deg: The image degraded
        :return:
        """
        x1, x2, y1, y2 = self.ctx.tile_position
        tile_mask = self.sph_points_mask_dict[self.projection][y1:y2, x1:x2]

        tile_ref_m = tile_ref * tile_mask
        tile_deg_m = tile_deg * tile_mask

        sqr_dif = (tile_ref_m - tile_deg_m) ** 2

        smse_nn = sqr_dif.sum() / np.sum(tile_mask)
        return smse_nn

    def make_sph_points_mask_dict(self) -> dict[str, np.ndarray]:
        proj_str = ', '.join(self.projection_list)
        sph_points_mask_file = Path(f'datasets/masks/sph_points_mask_{proj_str}.pickle')
        sph_file = Path('datasets/sphere_655362.txt')

        if sph_points_mask_file.exists():
            return load_pickle(sph_points_mask_file)

        sph_points_mask = {self.projection: self.process_sphere_file(sph_file, self.projection)
                           for self.projection in self.projection_list}
        save_pickle(sph_points_mask, sph_points_mask_file)
        return sph_points_mask

    def process_sphere_file(self, sph_file, proj) -> np.ndarray:
        """
        O arquivo txt contem as coordenadas como ea (elevação, azimute).
        Vamos identificar pontos na esfera (r=1) com essas coordenadas e marcar nas
        faces da projeção estes pontos. A projeção deve ter a mesma resolução dos
        vídeos, (2160, 3240) para CMP. Este mapa é guardado e usado como máscara
        para selecionar quais pixeis devem ser considerados no cálculo do s-mse.

        :return:
        """
        sph_file_lines = sph_file.read_text().splitlines()[1:]
        ea_array = np.array(list(map(lines_2_list, sph_file_lines))).T

        if proj == 'cmp':
            nm = CMP.ea2nm_face(ea=ea_array, proj_shape=self.video_shape)[0]
        elif proj == 'erp':
            nm = ERP.ea2nm(ea=ea_array, proj_shape=self.video_shape)
        else:
            nm = np.ndarray([])

        sph_points_mask = np.zeros(self.video_shape)
        sph_points_mask[nm[0], nm[1]] = 1
        return sph_points_mask


def make_sph_points_mask_dict(ctx: Context) -> np.ndarray:
    """
    Load 655362 sample points (elevation, azimuth). Angles in degree.

    :return:
    """
    sph_points_mask_file = Path(f'datasets/masks/sph_points_mask.pickle')

    try:
        sph_points_mask = load_pickle(sph_points_mask_file)
    except FileNotFoundError:
        sph_points_mask = process_sphere_file(ctx)
        save_pickle(sph_points_mask, sph_points_mask_file)
    return sph_points_mask


def process_sphere_file(ctx: Context) -> dict[str, np.ndarray]:
    """
    O arquivo txt contem as coordenadas como ea (elevação, azimute).
    Vamos identificar pontos na esfera (r=1) com essas coordenadas e marcar nas
    faces da projeção estes pontos. A projeção deve ter a mesma resolução dos
    vídeos, (2160, 3240) para CMP. Este mapa é guardado e usado como máscara
    para selecionar quais pixeis devem ser considerados no cálculo do s-mse.


    :param ctx:
    :return:
    """
    sph_file = Path('datasets/sphere_655362.txt')
    sph_file_lines = sph_file.read_text().splitlines()[1:]
    ea_array = np.array(list(map(lines_2_list, sph_file_lines))).T

    sph_points_mask = {}
    for proj in ctx.projection_list:
        proj_shape = splitx(ctx.config.config_dict['scale'][proj])[::-1]
        if proj == 'erp':
            nm = ERP.ea2nm(ea=ea_array, proj_shape=proj_shape)
        elif proj == 'cmp':
            nm = CMP.ea2nm_face(ea=ea_array, proj_shape=proj_shape)[0]
        else:
            raise ValueError(f'Unknown projection: {proj}')
        sph_points_mask[proj] = np.zeros(proj_shape)
        sph_points_mask[proj][nm[0], nm[1]] = 1

    return sph_points_mask


def lines_2_list(line) -> list:
    strip_line = line.strip()
    split_line = strip_line.split()
    map_float_line = map(float, split_line)
    map_rad_line = map(np.deg2rad, map_float_line)
    return list(map_rad_line)


def make_weight_ndarray_dict(ctx: Context) -> dict[str, np.ndarray]:
    project_folder: Path = ctx.config.project_folder
    weight_ndarray_dict_file = Path(f'datasets/{project_folder.name}/weight_ndarray_dict.pickle')

    try:
        weight_ndarray_dict = load_pickle(weight_ndarray_dict_file)
    except FileNotFoundError:
        weight_ndarray_dict = process_weight_ndarray_dict_file(ctx)
        save_pickle(weight_ndarray_dict, weight_ndarray_dict_file)
    return weight_ndarray_dict


def process_weight_ndarray_dict_file(ctx: Context) -> dict[str, np.ndarray]:
    weight_array = {}
    for proj in ctx.projection_list:
        scale = ctx.config.config_dict['scale'][proj]
        w, h = splitx(scale)

        if proj == 'erp':
            pi_proj = np.pi / h
            proj_h_2 = 0.5 - h / 2

            def func(y, x):
                weights = np.cos((y + proj_h_2) * pi_proj)
                return weights

        elif proj == 'cmp':
            a = h / 2
            r = a / 2
            r1 = 0.5 - r
            r2 = r ** 2

            def func(y, x):
                x = x % a
                y = y % a
                d = (x + r1) ** 2 + (y + r1) ** 2
                weights = (1 + d / r2) ** (-1.5)
                return weights
        else:
            raise ValueError(f'Unknown projection: {proj}')
        weight_array[proj] = np.fromfunction(func, (h, w), dtype=float)
    # plt.imshow(np.ones((cmp_h, cmp_w)) * 255 * np.fromfunction(func_cmp, (cmp_h, cmp_w), dtype=float));plt.show()
    # plt.imshow(np.ones((erp_h, erp_w)) * 255 * np.fromfunction(func_erp, (erp_h, erp_w), dtype=float));plt.show()
    return weight_array
