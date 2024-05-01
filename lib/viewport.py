from typing import Union, Callable, Optional

import cv2
import numpy as np

from .util import get_borders, rot_matrix


class ViewportProps:
    base_normals: np.ndarray
    fov: np.ndarray
    vp_xyz_base: np.ndarray
    vp_shape: np.ndarray
    base_vp_xyz: np.ndarray

    _yaw_pitch_roll: np.ndarray

    _mat_rot: Optional[np.ndarray] = None
    _normals_rotated: Optional[np.ndarray] = None
    _vp_rotated_xyz: Optional[np.ndarray] = None
    _vp_img: Optional[np.ndarray] = None

    @property
    def yaw_pitch_roll(self):
        return self._yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value):
        self._yaw_pitch_roll = value
        self._mat_rot = None
        self._normals_rotated = None
        self._vp_rotated_xyz = None
        self._vp_img = None

    @property
    def mat_rot(self) -> np.ndarray:
        if self._mat_rot is not None:
            return self._mat_rot

        self._mat_rot = rot_matrix(self.yaw_pitch_roll)
        return self._mat_rot

    @property
    def rotated_normals(self) -> np.ndarray:
        if self._normals_rotated is not None:
            return self._normals_rotated

        self._normals_rotated = np.tensordot(self.mat_rot, self.base_normals, axes=1)
        return self._normals_rotated

    @property
    def vp_xyz_rotated(self) -> np.ndarray:
        if self._vp_rotated_xyz is not None:
            return self._vp_rotated_xyz

        self._vp_rotated_xyz = np.tensordot(self.mat_rot, self.base_vp_xyz, axes=1)
        return self._vp_rotated_xyz


class Viewport(ViewportProps):
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
        self._make_normals_base()
        self._make_vp_xyz_base()
        self.yaw_pitch_roll = np.array([0, 0, 0])

    def _make_normals_base(self) -> None:
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
        fov_2 = self.fov / (2, 2)
        cos_fov = np.cos(fov_2)
        sin_fov = np.sin(fov_2)

        self.base_normals = np.array([[0, -cos_fov[0], -sin_fov[0]],  # top
                                      [0, cos_fov[0], -sin_fov[0]],  # bottom
                                      [-cos_fov[1], 0, -sin_fov[1]],  # left
                                      [cos_fov[1], 0, -sin_fov[1]]]).T  # right

    def _make_vp_xyz_base(self) -> None:
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
        y_coord = np.linspace(-tan_fov_2[0], tan_fov_2[0], self.vp_shape[0], endpoint=True)
        x_coord = np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_shape[1], endpoint=False)

        vp_coord_x, vp_coord_y = np.meshgrid(x_coord, y_coord)
        vp_coord_z = np.ones(self.vp_shape)
        vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z])

        r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=0, keepdims=True))
        # np.power
        self.base_vp_xyz = vp_coord_xyz_ / r  # normalize. final shape==(3,H,W)

    def is_viewport(self, x_y_z: np.ndarray) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x_y_z: A 3D Point list in the space [(x, y, z), ...].T, shape == (3, ...)
        :return: A boolean         belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)

        """
        inner_prod = np.tensordot(self.rotated_normals.T, x_y_z, axes=1)
        belong = np.all(inner_prod <= 0, axis=0)
        is_vp = np.any(belong)
        return is_vp

    def get_vp(self, frame: np.ndarray, xyz2nm: Callable) -> np.ndarray:
        """

        :param frame: The projection image. (N,M,C)
        :param xyz2nm: A function from 3D to projection.
        :return: The viewport image (RGB)
        """
        if self._vp_img is not None:
            return self._vp_img

        nm_coord = xyz2nm(self.vp_xyz_rotated, frame.shape[:2])
        nm_coord = nm_coord.transpose((1, 2, 0))
        self._vp_img = cv2.remap(frame,
                                 map1=nm_coord[..., 1:2].astype(np.float32),
                                 map2=nm_coord[..., 0:1].astype(np.float32),
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_WRAP)
        # show1(self._out)
        return self._vp_img

    _vp_borders_xyz: np.ndarray

    def get_vp_borders_xyz(self, thickness: int = 1) -> np.ndarray:
        """

        :param thickness: in pixels
        :return: np.ndarray (shape == (1,HxW,3)
        """
        if self._vp_borders_xyz:
            return self._vp_borders_xyz

        self._vp_borders_xyz = get_borders(coord_nm=self.vp_xyz_rotated, thickness=thickness)
        return self._vp_borders_xyz
