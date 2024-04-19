from abc import ABC, abstractmethod
from math import pi
from typing import Union, Callable, Optional

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

try:
    from .transform import (rot_matrix, splitx, cmp2nmface, nmface2vuface, vuface2xyz_face, xyz2vuface, vuface2nmface, nmface2cmp_face)
except ImportError:
    from lib.transform import (rot_matrix, splitx, cmp2nmface, nmface2vuface, vuface2xyz_face, xyz2vuface, vuface2nmface, nmface2cmp_face)


class Viewport:
    base_normals: np.ndarray
    fov: np.ndarray
    vp_state: set
    vp_coord_xyz: np.ndarray
    vp_shape: np.ndarray

    _yaw_pitch_roll: np.ndarray

    def __init__(self, vp_shape: Union[np.ndarray, tuple], fov: np.ndarray):
        """
        Viewport Class used to extract view pixels in projections.
        The vp is an image as numpy array with shape (H, M, 3).
        That can be RGB (matplotlib, pillow, etc.) or BGR (opencv).

        :param frame vp_shape: (600, 800) for 800x600px
        :param fov: in rad. Ex: "np.array((pi/2, pi/2))" for (90°x90°)
        """
        self.fov = fov
        self.vp_shape = vp_shape
        self.vp_state = set()
        self._make_base_normals()
        self._make_base_vp_coord()

        self._yaw_pitch_roll = np.array([0, 0, 0])

    def _make_base_normals(self) -> None:
        """
        Com eixo entrando no observador, rotação horário é negativo e anti-horária
        é positivo. Todos os ângulos são radianos.

        O eixo x aponta para a direita
        O eixo y aponta para baixo
        O eixo z aponta para a frente

        Deslocamento para a direita e para cima é positivo.

        O viewport é a região da esfera que faz intersecção com 4 planos que passam pelo
          centro (4 grandes círculos): cima, direita, baixo e esquerda.
        Os planos são definidos tal que suas normais (N) parte do centro e apontam na mesma direção a
          região do viewport. Ex: O plano de cima aponta para cima, etc.
        Todos os píxeis que estiverem abaixo do plano {N(x,y,z) dot P(x,y,z) <= 0}
        O plano de cima possui inclinação de FOV_Y / 2.
          Sua normal é x=0,y=sin(FOV_Y/2 + pi/2), z=cos(FOV_Y/2 + pi/2)
        O plano de baixo possui inclinação de -FOV_Y / 2.
          Sua normal é x=0,y=sin(-FOV_Y/2 - pi/2), z=cos(-FOV_Y/2 - pi/2)
        O plano da direita possui inclinação de FOV_X / 2. (para direita)
          Sua normal é x=sin(FOV_X/2 + pi/2),y=0, z=cos(FOV_X/2 + pi/2)
        O plano da esquerda possui inclinação de -FOV_X/2. (para direita)
          Sua normal é x=sin(-FOV_X/2 - pi/2),y=0, z=cos(-FOV_X/2 - pi/2)

        :return:
        """
        fov_y_2, fov_x_2 = self.fov / (2, 2)
        pi_2 = np.pi / 2

        self.base_normals = np.array([[0, -np.sin(fov_y_2 + pi_2), np.cos(fov_y_2 + pi_2)],  # top
                                      [0, -np.sin(-fov_y_2 - pi_2), np.cos(-fov_y_2 - pi_2)],  # bottom
                                      [np.sin(fov_x_2 + pi_2), 0, np.cos(fov_x_2 + pi_2)],  # left
                                      [np.sin(-fov_x_2 - pi_2), 0, np.cos(-fov_x_2 - pi_2)]]).T  # right

    def _make_base_vp_coord(self) -> None:
        """
        The VP projection is based in rectilinear projection.

        In the sphere domain, in te cartesian system, the center of a plain touch the sphere
        on the point (x=0,y=0,z=1).
        The plain sizes are based on the tangent of fov.
        The resolution (number of samples) of viewport is defined by the constructor.
        The proj_coord_xyz.shape is (3,H,W). The dim 0 are x, y z coordinates.
        :return:
        """
        tan_fov_2 = np.tan(self.fov / 2)
        x_coord = np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_shape[1], endpoint=False)
        y_coord = np.linspace(-tan_fov_2[0], tan_fov_2[0], self.vp_shape[0], endpoint=True)

        (vp_coord_x, vp_coord_y), vp_coord_z = np.meshgrid(x_coord, y_coord), np.ones(self.vp_shape)
        vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z])

        r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=0, keepdims=True))

        self.vp_coord_xyz = vp_coord_xyz_ / r  # normalize. final shape==(3,H,W)

    _is_viewport: Optional[bool] = None

    def is_viewport(self, x_y_z: np.ndarray) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x_y_z: A 3D Point list in the space [(x, y, z), ...].T, shape == (3, ...)
        :return: A boolean         belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)

        """
        if self._is_viewport is not None:
            return self._is_viewport

        inner_prod = self.rotated_normals.T @ x_y_z
        px_in_vp = np.all(inner_prod <= 0, axis=0)
        self._is_viewport = np.any(px_in_vp)
        return self._is_viewport

    _vp_img = None

    def get_vp(self, frame: np.ndarray, xyz2nm: Callable) -> np.ndarray:
        """

        :param frame: The projection image.
        :param xyz2nm: A function from 3D to projection.
        :return: The viewport image (RGB)
        """
        if self._vp_img is not None:
            return self._vp_img

        nm_coord = xyz2nm(self.vp_rotated_xyz, frame.shape)
        nm_coord = nm_coord.transpose((1, 2, 0))
        self._vp_img = cv2.remap(frame, map1=nm_coord[..., 1:2].astype(np.float32), map2=nm_coord[..., 0:1].astype(np.float32), interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_WRAP)
        # show2(self._out)
        return self._vp_img

    # def get_vp_borders_xyz(self, thickness: int = 1) -> np.ndarray:
    #     """
    #
    #     :param thickness: in pixels
    #     :return: np.ndarray (shape == (1,HxW,3)
    #     """
    #     if self._vp_borders_xyz:
    #         return self._vp_borders_xyz
    #
    #     self._vp_borders_xyz = get_borders(coord_nm=self.vp_rotated_xyz, thickness=thickness)
    #     return self._vp_borders_xyz

    @property
    def yaw_pitch_roll(self) -> np.ndarray:
        return self._yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value: np.ndarray):
        """
        Set a new position to viewport using aerospace's body coordinate system
        and rotate the normals. Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Y-X-Z order. Refer to Wikipedia.

        :param value: the positions like array(yaw, pitch, roll) in rad
        """
        self._yaw_pitch_roll = value
        self._vp_rotated_xyz = None
        self._mat_rot = None
        self._rotated_normals = None
        self._vp_img = None
        # self._vp_borders_xyz = None
        self._is_viewport = None

    _vp_rotated_xyz = None

    @property
    def vp_rotated_xyz(self) -> np.ndarray:
        if self._vp_rotated_xyz is not None:
            return self._vp_rotated_xyz

        self._vp_rotated_xyz = np.tensordot(self.mat_rot, self.vp_coord_xyz, axes=1)
        return self._vp_rotated_xyz

    _mat_rot = None

    @property
    def mat_rot(self) -> np.ndarray:
        if self._mat_rot is not None:
            return self._mat_rot

        self._mat_rot = rot_matrix(self.yaw_pitch_roll)
        return self._mat_rot

    _rotated_normals = None

    @property
    def rotated_normals(self) -> np.ndarray:
        if self._rotated_normals is not None:
            return self._rotated_normals

        self._rotated_normals = self.mat_rot @ self.base_normals
        return self._rotated_normals


class ProjProps(ABC):
    proj_res: str
    tiling: str
    fov: str
    vp_rotated_xyz: np.ndarray
    frame_img = np.zeros([0])

    @abstractmethod
    def nm2xyz(self, nm_coord: np.ndarray, shape: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def xyz2nm(self, xyz_coord: np.ndarray, shape: Union[np.ndarray, tuple], round_nm: bool) -> np.ndarray:
        pass

    # <editor-fold desc="About the Projection">
    _proj_shape: np.ndarray = None

    @property
    def proj_shape(self) -> np.ndarray:
        if self._proj_shape is None:
            self._proj_shape = np.array(splitx(self.proj_res)[::-1], dtype=int)
        return self._proj_shape

    _proj_h: int = None

    @property
    def proj_h(self) -> int:
        if self._proj_h is None:
            self._proj_h = self.proj_shape[0]
        return self._proj_h

    _proj_w: int = None

    @property
    def proj_w(self) -> int:
        if self._proj_w is None:
            self._proj_w = self.proj_shape[1]
        return self._proj_w

    _projection: np.ndarray = None

    @property
    def projection(self) -> np.ndarray:
        if self._projection is None:
            self._projection = np.zeros(self.proj_shape, dtype='uint8')
        return self._projection

    @projection.setter
    def projection(self, value):
        self._projection = value

    _proj_coord_nm: Union[np.ndarray, list] = None

    @property
    def proj_coord_nm(self) -> np.ndarray:
        if self._proj_coord_nm is None:
            self._proj_coord_nm = np.mgrid[0:self.proj_h, 0:self.proj_w]
        return self._proj_coord_nm

    _proj_coord_xyz: np.ndarray = None

    @property
    def proj_coord_xyz(self) -> np.ndarray:
        if self._proj_coord_xyz is None:
            self._proj_coord_xyz = self.nm2xyz(self.proj_coord_nm, self.proj_shape)
        return self._proj_coord_xyz

    # </editor-fold>

    # <editor-fold desc="About Tiling">
    _tiling_shape: np.ndarray = None

    @property
    def tiling_shape(self) -> np.ndarray:
        if self._tiling_shape is None:
            self._tiling_shape = np.array(splitx(self.tiling)[::-1], dtype=int)
        return self._tiling_shape

    _tiling_h: int = None

    @property
    def tiling_h(self) -> int:
        if not self._tiling_h:
            self._tiling_h = self.tiling_shape[0]
        return self._tiling_h

    _tiling_w: int = None

    @property
    def tiling_w(self) -> int:
        if not self.tiling_w:
            self._tiling_w = self.tiling_shape[1]
        return self.tiling_w

    # </editor-fold>

    # <editor-fold desc="About Tiles">
    _n_tiles: int = None

    @property
    def n_tiles(self) -> int:
        if not self._n_tiles:
            self._n_tiles = self.tiling_shape[0] * self.tiling_shape[1]
        return self._n_tiles

    _tile_shape: np.ndarray = None

    @property
    def tile_shape(self) -> np.ndarray:
        if self._tile_shape is None:
            self._tile_shape = (self.proj_shape / self.tiling_shape).astype(int)
        return self._tile_shape

    _tile_h: int = None

    @property
    def tile_h(self) -> int:
        if not self._tile_h:
            self._tile_h = self.tile_shape[0]
        return self._tile_h

    _tile_w: int = None

    @property
    def tile_w(self) -> int:
        if not self._tile_w:
            self._tile_w = self.tile_shape[1]
        return self._tile_w

    _tile_position_list: np.ndarray = None

    @property
    def tile_position_list(self) -> np.ndarray:
        """
        top-left pixel position
        :return: (N,2)
        """
        if self._tile_position_list is None:
            tile_position_list = []
            for n in range(0, self.proj_shape[0], self.tile_shape[0]):
                for m in range(0, self.proj_shape[1], self.tile_shape[1]):
                    tile_position_list.append((n, m))
            self._tile_position_list = np.array(tile_position_list)
        return self._tile_position_list

    _tile_border_base: np.ndarray = None

    @property
    def tile_border_base(self) -> np.ndarray:
        """

        :return: shape==(2, 2*(tile_height+tile_weight)
        """
        if self._tile_border_base is None:
            self._tile_border_base = get_borders(shape=self.tile_shape)
        return self._tile_border_base

    _tile_borders_nm: np.ndarray = None

    @property
    def tile_borders_nm(self) -> np.ndarray:
        """

        :return:
        """
        # projection agnostic
        if self._tile_borders_nm is None:
            _tile_borders_nm = []
            for tile in range(self.n_tiles):
                tile_position = self.tile_position_list[tile].reshape(2, -1)
                _tile_borders_nm.append(self.tile_border_base + tile_position)
            self._tile_borders_nm = np.array(_tile_borders_nm)
        return self._tile_borders_nm

    _tile_borders_xyz: list = None

    @property
    def tile_borders_xyz(self) -> list:
        """
        shape = (3, H, W) "WxH array" OR (3, N) "N points (z, y, x)"
        :return:
        """
        if not self._tile_borders_xyz:
            self._tile_borders_xyz = []
            for tile in range(self.n_tiles):
                borders_nm = self.tile_borders_nm[tile]
                borders_xyz = self.nm2xyz(nm_coord=borders_nm, shape=self.proj_shape)
                self._tile_borders_xyz.append(borders_xyz)
        return self._tile_borders_xyz

    # </editor-fold>

    # <editor-fold desc="About Viewport">
    _fov_shape: np.ndarray = None

    @property
    def fov_shape(self) -> np.ndarray:
        if self._fov_shape is None:
            self._fov_shape = np.deg2rad(splitx(self.fov)[::-1])
        return self._fov_shape

    _vp_shape: np.ndarray = None

    @property
    def vp_shape(self) -> np.ndarray:
        if self._vp_shape is None:
            self._vp_shape = np.round(self.fov_shape * self.proj_shape / (pi, 2 * pi)).astype('int')
        return self._vp_shape

    @vp_shape.setter
    def vp_shape(self, value: np.ndarray):
        self._vp_shape = value

    _viewport: Viewport = None

    @property
    def viewport(self) -> Viewport:
        if not self._viewport:
            self._viewport = Viewport(self.vp_shape, self.fov_shape)
        return self._viewport

    _vp_image: np.ndarray = None

    @property
    def vp_image(self) -> np.ndarray:
        nm_coord = self.xyz2nm(self.viewport.vp_rotated_xyz, self.frame_img.shape, round_nm=False)
        nm_coord = nm_coord.transpose((1, 2, 0))
        out = cv2.remap(self.frame_img, map1=nm_coord[..., 1:2].astype(np.float32), map2=nm_coord[..., 0:1].astype(np.float32), interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)

        self._vp_image = out
        return self._vp_image

    # </editor-fold>

    # <editor-fold desc="About Position">
    @property
    def yaw_pitch_roll(self) -> np.ndarray:
        return self.viewport.yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value: Union[np.ndarray, list]):
        self.viewport.yaw_pitch_roll = np.array(value)

    # </editor-fold>


class ProjBase(ProjProps, ABC):
    def __init__(self, *, proj_res: str, tiling: str, fov: str, vp_shape: np.ndarray = None):
        self.proj_res = proj_res
        self.tiling = tiling
        self.fov = fov
        self.vp_shape = vp_shape
        self.yaw_pitch_roll = [0, 0, 0]

    @abstractmethod
    def nm2xyz(self, nm_coord: np.ndarray, shape: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def xyz2nm(self, xyz_coord: np.ndarray, proj_shape: Union[np.ndarray, tuple], round_nm: bool) -> np.ndarray:
        pass

    def get_vptiles(self) -> list[str]:
        """

        :return:
        """
        if self.tiling == '1x1': return ['0']
        vptiles = [str(tile) for tile in range(self.n_tiles) if self.viewport.is_viewport(self.tile_borders_xyz[tile])]
        return vptiles

    def get_viewport(self, frame_img: np.ndarray, yaw_pitch_roll=None) -> np.ndarray:
        if yaw_pitch_roll is not None:
            self.yaw_pitch_roll = yaw_pitch_roll
        self.frame_img = frame_img

        nm_coord = self.xyz2nm(self.viewport.vp_rotated_xyz, self.frame_img.shape, round_nm=False)
        nm_coord = nm_coord.transpose((1, 2, 0))
        out = cv2.remap(self.frame_img, map1=nm_coord[..., 1:2].astype(np.float32), map2=nm_coord[..., 0:1].astype(np.float32), interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)

        return out

    def show(self):
        show1(self.projection)

    ##############################################
    # Draw methods
    def draw_all_tiles_borders(self, lum=255):
        self.clear_projection()
        for tile in range(self.n_tiles):
            self.draw_tile_border(idx=int(tile), lum=lum)
        return self.projection

    def draw_vp_tiles(self, lum=255):
        self.clear_projection()
        for tile in self.get_vptiles():
            self.draw_tile_border(idx=int(tile), lum=lum)
        return self.projection

    def draw_tile_border(self, idx, lum=255):
        n, m = self.tile_borders_nm[idx]
        self.projection[n, m] = lum

    def draw_vp_mask(self, lum=255) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        self.clear_projection()
        rotated_normals = self.viewport.rotated_normals.T
        inner_product = np.tensordot(rotated_normals, self.proj_coord_xyz, axes=1)
        belong = np.all(inner_product <= 0, axis=0)
        self.projection[belong] = lum
        return self.projection

    def draw_vp_borders(self, lum=255, thickness=1):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :param thickness: in pixel.
        :return: a numpy.ndarray with one deep color
        """
        self.clear_projection()

        vp_borders_xyz = get_borders(coord_nm=self.viewport.vp_rotated_xyz, thickness=thickness)

        nm_coord = self.xyz2nm(vp_borders_xyz, proj_shape=self.proj_shape, round_nm=True).astype(int)
        self.projection[nm_coord[0, ...], nm_coord[1, ...]] = lum
        return self.projection

    def clear_projection(self):
        self.projection = None


class ERP(ProjBase):
    def nm2xyz(self, nm_coord: np.ndarray, proj_shape: np.ndarray):
        """
        ERP specific.

        :param nm_coord: shape==(2,...)
        :param proj_shape: (N, M)
        :return:
        """
        azimuth = ((nm_coord[1] + 0.5) / proj_shape[1] - 0.5) * 2 * np.pi
        elevation = ((nm_coord[0] + 0.5) / proj_shape[0] - 0.5) * -np.pi

        z = np.cos(elevation) * np.cos(azimuth)
        y = -np.sin(elevation)
        x = np.cos(elevation) * np.sin(azimuth)

        xyz_coord = np.array([x, y, z])
        return xyz_coord

    def xyz2nm(self, xyz_coord: np.ndarray, proj_shape: np.ndarray = None, round_nm: bool = False):
        """
        ERP specific.

        :param xyz_coord: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param proj_shape: the shape of projection that cover all sphere
        :param round_nm: round the coords? is not needed.
        :return:
        """
        if proj_shape is None:
            proj_shape = xyz_coord.shape[:2]

        proj_h, proj_w = proj_shape[:2]

        r = np.sqrt(np.sum(xyz_coord ** 2, axis=0))

        elevation = np.arcsin(xyz_coord[1] / r)
        azimuth = np.arctan2(xyz_coord[0], xyz_coord[2])

        v = elevation / pi + 0.5
        u = azimuth / (2 * pi) + 0.5

        n = v * proj_h - 0.5
        m = u * proj_w - 0.5

        if round_nm:
            n = np.mod(np.round(n), proj_h)
            m = np.mod(np.round(m), proj_w)

        return np.array([n, m])


class CMP(ProjBase):
    def nm2xyz(self, nm_coord: np.ndarray, shape: Union[np.ndarray, tuple], rotate: bool = True):
        """
        CMP specific.

        :param nm_coord: shape==(2,...)
        :param shape: (N, M)
        :param rotate: True
        :return: x, y, z
        """
        nmface = cmp2nmface(nm_coord, shape)
        vuface = nmface2vuface(nmface)
        xyz, face = vuface2xyz_face(vuface)
        return xyz

    def xyz2nm(self, xyz_coord: np.ndarray, proj_shape: np.ndarray = None, round_nm: bool = False, rotate=True):
        """
        CMP specific.

        :param rotate:
        :param xyz_coord: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param proj_shape: the shape of projection that cover all sphere
        :param round_nm: round the coords? is not needed.
        :return:
        """
        vuface = xyz2vuface(xyz_coord)
        nmface = vuface2nmface(vuface, proj_shape=proj_shape)
        cmp, face = nmface2cmp_face(nmface, proj_shape=proj_shape)

        # fig = plt.figure()
        #
        # ax = fig.add_subplot(projection='3d')
        #
        # ax.scatter(0, 0, 0, marker='o', color='red')
        # ax.scatter(1, 1, 1, marker='o', color='red')
        # ax.scatter(1, 1, -1, marker='o', color='red')
        # ax.scatter(1, -1, 1, marker='o', color='red')
        # ax.scatter(1, -1, -1, marker='o', color='red')
        # ax.scatter(-1, 1, 1, marker='o', color='red')
        # ax.scatter(-1, 1, -1, marker='o', color='red')
        # ax.scatter(-1, -1, 1, marker='o', color='red')
        # ax.scatter(-1, -1, -1, marker='o', color='red')
        # [ax.scatter(x, y, z, marker='o', color='red') for x, y, z in zip(xyz_coord[0, 0:4140:100], xyz_coord[1, 0:4140:100], xyz_coord[2, 0:4140:100])]
        #
        # face0 = vuface[2] == 0
        # face1 = vuface[2] == 1
        # face2 = vuface[2] == 2
        # face3 = vuface[2] == 3
        # face4 = vuface[2] == 4
        # face5 = vuface[2] == 5
        # [ax.scatter(-1, v, u, marker='o', color='blue') for v, u in zip(vuface[0, face0][::25], vuface[1, face0][::25])]
        # [ax.scatter(u, v, 1, marker='o', color='blue') for v, u in zip(vuface[0, face1][::25], vuface[1, face1][::25])]
        # [ax.scatter(1, v, -u, marker='o', color='blue') for v, u in zip(vuface[0, face2][::25], vuface[1, face2][::25])]
        # [ax.scatter(-u, 1, v, marker='o', color='blue') for v, u in zip(vuface[0, face3][::25], vuface[1, face3][::25])]
        # [ax.scatter(-u, v, -1, marker='o', color='blue') for v, u in zip(vuface[0, face4][::25], vuface[1, face4][::25])]
        # [ax.scatter(-u, -1, 1, marker='o', color='blue') for v, u in zip(vuface[0, face5][::25], vuface[1, face5][::25])]
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

        return cmp


def show1(projection: np.ndarray):
    plt.imshow(projection)
    plt.show()


def show2(projection: np.ndarray):
    frame_img = Image.fromarray(projection)
    frame_img.show()


def get_borders(*, coord_nm: Union[tuple, np.ndarray] = None, shape=None, thickness=1):
    """
    coord_nm must be shape==(C, N, M)
    :param coord_nm:
    :param shape:
    :param thickness:
    :return: shape==(C, thickness*(2N+2M))
    """
    if coord_nm is None:
        assert shape is not None
        coord_nm = np.mgrid[0:shape[0], 0:shape[1]]
        c = 2
    else:
        c = coord_nm.shape[0]

    left = coord_nm[:, :, 0:thickness].reshape((c, -1))
    right = coord_nm[:, :, :- 1 - thickness:-1].reshape((c, -1))
    top = coord_nm[:, 0:thickness, :].reshape((c, -1))
    bottom = coord_nm[:, :- 1 - thickness:-1, :].reshape((c, -1))

    return np.c_[top, right, bottom, left]


def compose(proj: ProjBase, proj_frame_array: Image) -> np.ndarray:
    tiles = proj.get_vptiles()
    frame_array = np.asarray(proj_frame_array)

    height, width, _ = frame_array.shape
    viewport_array = proj.get_viewport(frame_array)
    vp_image = Image.fromarray(viewport_array)
    width_vp = int(np.round(height * vp_image.width / vp_image.height))
    vp_image_resized = vp_image.resize((width_vp, height))

    cover_red = Image.new("RGB", (width, height), (255, 0, 0))
    cover_green = Image.new("RGB", (width, height), (0, 255, 0))
    cover_gray = Image.new("RGB", (width, height), (200, 200, 200))
    cover_blue = Image.new("RGB", (width, height), (0, 0, 255))

    # Get masks
    mask_all_tiles_borders = Image.fromarray(proj.draw_all_tiles_borders())
    mask_vp_tiles = Image.fromarray(proj.draw_vp_tiles())
    mask_vp = Image.fromarray(proj.draw_vp_mask(lum=200))
    mask_vp_borders = Image.fromarray(proj.draw_vp_borders())

    # Composite mask with projection
    proj_frame_array = Image.composite(cover_red, proj_frame_array, mask=mask_all_tiles_borders)
    proj_frame_array = Image.composite(cover_green, proj_frame_array, mask=mask_vp_tiles)
    proj_frame_array = Image.composite(cover_gray, proj_frame_array, mask=mask_vp)
    proj_frame_array = Image.composite(cover_blue, proj_frame_array, mask=mask_vp_borders)

    new_im = Image.new('RGB', (width + width_vp + 2, height), (255, 255, 255))
    new_im.paste(proj_frame_array, (0, 0))
    new_im.paste(vp_image_resized, (width + 2, 0))
    new_im.show()

    print(f'The viewport touch the tiles {tiles}.')
    return np.asarray(new_im)


def test_erp():
    # erp '144x72', '288x144','432x216','576x288'
    # cmp '144x96', '288x192','432x288','576x384'
    yaw_pitch_roll = np.deg2rad((70, 0, 0))
    height, width = 288, 576

    ########################################
    # Open Image
    frame_img: Union[Image, list] = Image.open('images/erp1.jpg')
    frame_img = frame_img.resize((width, height))

    erp = ERP(tiling='6x4', proj_res=f'{width}x{height}', fov='100x90')
    erp.yaw_pitch_roll = yaw_pitch_roll
    compose(erp, frame_img)


def test_cmp():
    # erp '144x72', '288x144','432x216','576x288'
    # cmp '144x96', '288x192','432x288','576x384'
    yaw_pitch_roll = np.deg2rad((70, 0, 0))
    height, width = 384, 576

    cover_red = Image.new("RGB", (width, height), (255, 0, 0))
    cover_green = Image.new("RGB", (width, height), (0, 255, 0))
    cover_gray = Image.new("RGB", (width, height), (200, 200, 200))
    cover_blue = Image.new("RGB", (width, height), (0, 0, 255))

    ########################################
    # Open Image
    frame_img: Union[Image, list] = Image.open('images/cmp1.png')
    frame_img = frame_img.resize((width, height))
    frame_array = np.asarray(frame_img)

    cmp = CMP(tiling='6x4', proj_res=f'{width}x{height}', fov='110x90')
    cmp.yaw_pitch_roll = yaw_pitch_roll
    tiles = cmp.get_vptiles()

    viewport_array = cmp.get_viewport(frame_array)
    vp_image = Image.fromarray(viewport_array)
    width_vp = int(np.round(height * vp_image.width / vp_image.height))
    vp_image_resized = vp_image.resize((width_vp, height))

    # Get masks
    mask_all_tiles_borders = Image.fromarray(cmp.draw_all_tiles_borders())
    mask_vp_tiles = Image.fromarray(cmp.draw_vp_tiles())
    mask_vp = Image.fromarray(cmp.draw_vp_mask(lum=200))
    mask_vp_borders = Image.fromarray(cmp.draw_vp_borders())

    # Composite mask with projection
    frame_img = Image.composite(cover_red, frame_img, mask=mask_all_tiles_borders)
    frame_img = Image.composite(cover_green, frame_img, mask=mask_vp_tiles)
    frame_img = Image.composite(cover_gray, frame_img, mask=mask_vp)
    frame_img = Image.composite(cover_blue, frame_img, mask=mask_vp_borders)

    new_im = Image.new('RGB', (width + width_vp + 2, height), (255, 255, 255))
    new_im.paste(frame_img, (0, 0))
    new_im.paste(vp_image_resized, (width + 2, 0))
    new_im.show()
    print(f'The viewport touch the tiles {tiles}.')


if __name__ == '__main__':
    # test_erp()
    test_cmp()
