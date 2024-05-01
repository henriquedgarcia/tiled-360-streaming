from abc import ABC, abstractmethod
from typing import Union, Callable

import numpy as np
from PIL import Image

from .util import get_borders, show1, splitx
from .viewport import Viewport


class Attributes:
    canvas: np.ndarray
    frame_img = np.zeros([0])
    n_tiles: int
    nm2xyz: Callable
    proj_coord_xyz: np.ndarray
    proj_shape: np.ndarray
    tile_border_base: np.ndarray
    tile_borders_nm: np.ndarray
    tile_borders_xyz: list
    tile_position_list: list
    tile_shape: Union[np.ndarray, tuple]
    tiling: str
    viewport: Viewport
    xyz2nm: Callable
    yaw_pitch_roll: np.ndarray


class TransformMethods:
    @staticmethod
    def xyz2ea(xyz: np.ndarray) -> np.ndarray:
        """
        Convert from cartesian system to horizontal coordinate system in radians
        :param xyz: shape = (3, ...)
        :return: np.ndarray([azimuth, elevation]) - in rad. shape = (2, ...)
        """
        ea = np.zeros((2,) + xyz.shape[1:])

        r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)

        ea[0] = np.arcsin(-xyz[1] / r)
        ea[1] = np.arctan2(xyz[0], xyz[2])
        ea[1] = (ea[1] + np.pi) % (2*np.pi) - np.pi

        return ea

    @staticmethod
    def ea2xyz(ae: np.ndarray) -> np.ndarray:
        """
        Convert from horizontal coordinate system  in radians to cartesian system.
        ISO/IEC JTC1/SC29/WG11/N17197l: Algorithm descriptions of projection format conversion and video quality metrics in
        360Lib Version 5
        :param np.ndarray ae: In Rad. Shape == (2, ...)
        :return: (x, y, z)
        """
        new_shape = (3,) + ae.shape[1:]
        xyz = np.zeros(new_shape)
        xyz[0] = np.cos(ae[0]) * np.sin(ae[1])
        xyz[1] = -np.sin(ae[0])
        xyz[2] = np.cos(ae[0]) * np.cos(ae[1])
        xyz_r = np.round(xyz, 6)
        return xyz_r

    @staticmethod
    def normalize_ea(ea):
        _90_deg = np.pi / 2
        _180_deg = np.pi
        _360_deg = 2 * np.pi

        new_ea = np.zeros(ea.shape)
        new_ea[0] = -np.abs(np.abs(ea[0] + _90_deg) - _180_deg) + _90_deg
        new_ea[1] = (ea[1] + _180_deg) % _360_deg - _180_deg

        return new_ea


class ViewportMethods(TransformMethods, Attributes):
    def get_vp_image(self, frame_img: np.ndarray, yaw_pitch_roll=None) -> np.ndarray:
        if yaw_pitch_roll is not None:
            self.yaw_pitch_roll = yaw_pitch_roll

        out = self.viewport.get_vp(frame_img, self.xyz2nm)
        return out


class TilesMethods(TransformMethods, Attributes):
    def get_tile_position_list(self):
        tile_position_list = []
        for n in range(0, self.proj_shape[0], self.tile_shape[0]):
            for m in range(0, self.proj_shape[1], self.tile_shape[1]):
                tile_position_list.append((n, m))
        return np.array(tile_position_list)

    def get_tile_borders_nm(self):
        tile_borders_nm = []
        for tile in range(self.n_tiles):
            tile_position = self.tile_position_list[tile].reshape(2, -1)
            tile_borders_nm.append(self.tile_border_base + tile_position)
        return np.array(tile_borders_nm)

    def get_tile_borders_xyz(self):
        tile_borders_xyz = []
        for tile in range(self.n_tiles):
            borders_nm = self.tile_borders_nm[tile]
            borders_xyz = self.nm2xyz(nm=borders_nm, proj_shape=self.proj_shape)
            tile_borders_xyz.append(borders_xyz)
        return tile_borders_xyz

    def get_vptiles(self, yaw_pitch_roll=None) -> list[str]:
        """

        :return:
        """
        if self.tiling == '1x1': return ['0']

        if yaw_pitch_roll is not None:
            self.yaw_pitch_roll = yaw_pitch_roll

        vptiles = []
        for tile in range(self.n_tiles):
            if self.viewport.is_viewport(self.tile_borders_xyz[tile]):
                vptiles.append(str(tile))
        return vptiles


class DrawMethods(TilesMethods, Attributes):
    def clear_projection(self):
        self.canvas[...] = 0

    def show(self):
        show1(self.canvas)

    def draw_tile_border(self, idx, lum=255) -> np.ndarray:
        """
        Do not return copy
        :param idx:
        :param lum:
        :return:
        """
        n, m = self.tile_borders_nm[idx]
        self.canvas[n, m] = lum
        return self.canvas

    def draw_all_tiles_borders(self, lum=255):
        self.clear_projection()
        for tile in range(self.n_tiles):
            self.draw_tile_border(idx=tile, lum=lum)
        return self.canvas.copy()

    def draw_vp_tiles(self, lum=255):
        self.clear_projection()
        for tile in self.get_vptiles():
            self.draw_tile_border(idx=int(tile), lum=lum)
        return self.canvas.copy()

    def draw_vp_mask(self, lum=200) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        self.clear_projection()
        proj_coord_xyz = self.proj_coord_xyz
        rotated_normals = self.viewport.rotated_normals

        inner_prod = np.tensordot(rotated_normals.T, proj_coord_xyz, axes=1)
        belong = np.all(inner_prod <= 0, axis=0)
        self.canvas[belong] = lum

        # belong = np.all(inner_product <= 0, axis=0)
        # self.canvas[belong] = lum
        return self.canvas.copy()

    def draw_vp_borders(self, lum=255, thickness=1):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :param thickness: in pixel.
        :return: a numpy.ndarray with one deep color
        """

        self.clear_projection()
        vp_borders_xyz = get_borders(coord_nm=self.viewport.vp_xyz_rotated, thickness=thickness)
        nm = self.xyz2nm(vp_borders_xyz, proj_shape=self.proj_shape).astype(int)
        self.canvas[nm[0, ...], nm[1, ...]] = lum
        return self.canvas.copy()


class Props(Attributes):
    proj_coord_nm: Union[list, np.ndarray]

    @property
    def yaw_pitch_roll(self) -> np.ndarray:
        return self.viewport.yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value: Union[np.ndarray, list]):
        self.viewport.yaw_pitch_roll = np.array(value)


class ProjBase(Props,
               DrawMethods,
               TilesMethods,
               ViewportMethods,
               TransformMethods,
               ABC):
    def __init__(self, *, proj_res: str, tiling: str, fov: str, vp_shape: Union[np.ndarray, tuple, list] = None):
        # About projection
        self.proj_res = proj_res
        self.proj_shape = np.array(splitx(self.proj_res)[::-1], dtype=int)
        self.proj_h = self.proj_shape[0]
        self.proj_w = self.proj_shape[1]
        self.canvas = np.zeros(self.proj_shape, dtype='uint8')
        self.proj_coord_nm = np.mgrid[0:self.proj_h, 0:self.proj_w]
        self.proj_coord_xyz = self.nm2xyz(self.proj_coord_nm, self.proj_shape)

        # About Tiling
        self.tiling = tiling
        self.tiling_shape = np.array(splitx(self.tiling)[::-1], dtype=int)
        self.tiling_h = self.tiling_shape[0]
        self.tiling_w = self.tiling_shape[1]

        # About Tiles
        self.n_tiles = self.tiling_h * self.tiling_w
        self.tile_shape = (self.proj_shape / self.tiling_shape).astype(int)
        self.tile_h = self.tile_shape[0]
        self.tile_w = self.tile_shape[1]
        self.tile_position_list = self.get_tile_position_list()
        self.tile_border_base = get_borders(shape=self.tile_shape)
        self.tile_borders_nm = self.get_tile_borders_nm()
        self.tile_borders_xyz = self.get_tile_borders_xyz()

        # About Viewport
        self.fov = fov
        self.fov_shape = np.deg2rad(splitx(self.fov)[::-1])
        if vp_shape is None:
            vp_shape = np.round(self.fov_shape * self.proj_shape[0] / 4).astype('int')
        self.vp_shape = np.asarray(vp_shape)
        self.viewport = Viewport(self.vp_shape, self.fov_shape)

        self.yaw_pitch_roll = [0, 0, 0]

    @abstractmethod
    def nm2xyz(self, nm: np.ndarray, proj_shape: np.ndarray) -> np.ndarray:
        """
        Projection specific.

        :param nm: shape==(2,...)
        :param proj_shape:
        :return:
        """
        pass

    @abstractmethod
    def xyz2nm(self, xyz: np.ndarray, proj_shape: Union[np.ndarray, tuple]) -> np.ndarray:
        """
        Projection specific.

        :param xyz: shape==(2,...)
        :param proj_shape:
        :return:
        """
        pass


def compose(proj: ProjBase,
            proj_frame_image: Image,
            all_tiles_borders_image: Image,
            vp_tiles_image: Image,
            vp_mask_image: Image,
            vp_borders_image: Image,
            vp_image: Image,
            ) -> Image:
    height, width = proj_frame_image.height, proj_frame_image.width

    # Composite mask with projection
    cover_red = Image.new("RGB", (width, height), (255, 0, 0))
    proj_frame_image_c = Image.composite(cover_red, proj_frame_image, mask=all_tiles_borders_image)

    cover_green = Image.new("RGB", (width, height), (0, 255, 0))
    proj_frame_image_c = Image.composite(cover_green, proj_frame_image_c, mask=vp_tiles_image)

    cover_gray = Image.new("RGB", (width, height), (200, 200, 200))
    proj_frame_image_c = Image.composite(cover_gray, proj_frame_image_c, mask=vp_mask_image)

    cover_blue = Image.new("RGB", (width, height), (0, 0, 255))
    proj_frame_image_c = Image.composite(cover_blue, proj_frame_image_c, mask=vp_borders_image)

    # Resize Viewport
    width_vp = int(np.round(height * vp_image.width / vp_image.height))
    vp_image_resized = vp_image.resize((width_vp, height))

    # Compose new image
    new_im = Image.new('RGB', (width + width_vp + 2, height), (255, 255, 255))
    new_im.paste(proj_frame_image_c, (0, 0))
    new_im.paste(vp_image_resized, (width + 2, 0))

    # new_im.show()
    return new_im
