from pathlib import Path

import numpy as np
from PIL import Image
from py360tools import ERP, CMP
from py360tools.utils import splitx
from skimage.metrics import mean_squared_error, structural_similarity

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.utils.io_util import save_pickle, load_pickle


def show(array):
    Image.fromarray(array).show()


class QualityMetrics(CtxInterface):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.make_sph_points_mask_dict()
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
        x1, x2, y1, y2 = self.tile_position
        weight_tile = self.weight_ndarray[self.projection][y1:y2, x1:x2]

        wmse = np.sum(weight_tile * (im_ref - im_deg) ** 2) / np.sum(weight_tile)
        return wmse

    def smse_nn(self, tile_ref: np.ndarray, tile_deg: np.ndarray) -> float:
        """
        Calculate of S-PSNR between two images. All arrays must be on the same
        resolution.

        :param tile_ref: The original image
        :param tile_deg: The image degraded
        :return:
        """
        x1, x2, y1, y2 = self.tile_position
        tile_mask = self.sph_points_mask_dict[self.projection][y1:y2, x1:x2]

        tile_ref_m = tile_ref * tile_mask
        tile_deg_m = tile_deg * tile_mask

        sqr_dif = (tile_ref_m - tile_deg_m) ** 2

        smse_nn = sqr_dif.sum() / np.sum(tile_mask)
        return smse_nn

    sph_points_mask_dict: dict

    def make_sph_points_mask_dict(self):
        """
        Processes the spherical file to identify points on the unit sphere (r=1)
        based on given elevation and azimuth coordinates. Marks these points on
        the faces of the specified projection type (CMP or ERP). The projection
        resolution must match the one used in videos, for example, (2160, 3240)
        for CMP. The resulting map is saved and used as a mask to determine which
        pixels are taken into account during the calculation of the s-MSE metric.

        :param sph_file: The spherical file containing elevation and azimuth
                         coordinates to process
        :type sph_file: pathlib.Path
        :param proj: The type of projection to be used ('cmp' or 'erp')
        :type proj: str
        :return: None
        """
        """
        O arquivo txt contem as coordenadas como ea (elevação, azimute).
        Vamos identificar pontos na esfera (r=1) com essas coordenadas e marcar nas
        faces da projeção estes pontos. A projeção deve ter a mesma resolução dos
        vídeos, (2160, 3240) para CMP. Este mapa é guardado e usado como máscara
        para selecionar quais pixeis devem ser considerados no cálculo do s-mse.

        :return:
        """
        def load_sph_file() -> np.ndarray:
            """
            Converte o arquivo original para radianos e converte o txt em array.
            Salva o resultado no disco e usa ele sempre que necessário em vez do
            txt.

            :param sph_file:
            :return:
            """
            sph_file_array = self.config.sph_file.with_suffix('.pickle')
            try:
                array = load_pickle(sph_file_array)
                return array
            except FileNotFoundError:
                sph_file = self.config.sph_file
                sph_file_lines = sph_file.read_text().splitlines()[1:]
                array = np.array(list(map(self.lines_2_list, sph_file_lines))).T
                save_pickle(array, sph_file_array)
                return array

        # ea_array = load_sph_file()

    def load_sph_points_mask_dict(self):
        sph_file = self.config.sph_file
        sph_file_lines = sph_file.read_text().splitlines()[1:]
        ea_array = np.array(list(map(self.lines_2_list, sph_file_lines))).T
        self.sph_points_mask_dict = {}

        if self.projection == 'cmp':
            nm = CMP.ea2nm_face(ea=ea_array, proj_shape=self.video_shape)[0]
        elif self.projection == 'erp':
            nm = ERP.ea2nm(ea=ea_array, proj_shape=self.video_shape)[0]
        else:
            nm = np.ndarray([])
        sph_points_mask = np.zeros(self.video_shape)
        sph_points_mask[nm[0], nm[1]] = 1
        self.sph_points_mask_dict[self.projection] = sph_points_mask
        sph_points_mask_file.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(sph_points_mask, sph_points_mask_file)

    @staticmethod
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
