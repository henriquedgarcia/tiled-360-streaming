from typing import Union

import numpy as np
from PIL import Image

from .projectionbase import ProjBase, compose


class Methods:
    faces_list = []

    @staticmethod
    def cmp2nmface(nm: np.ndarray, proj_shape: tuple = None) -> np.ndarray:
        """

        :param proj_shape:
        :param nm: shape(2, ...)
                   pixel coords in image; n = height, m = width
        :return: nm_face(3, ...)
        """
        new_shape = (3,) + nm.shape[1:]
        nmface = np.zeros(new_shape)

        if proj_shape is None:
            proj_shape = nm.shape

        face_size = proj_shape[-1] // 3
        nmface[2] = nm[1] // face_size + (nm[0] // face_size) * 3

        face0 = nmface[2] == 0
        nmface[:2, face0] = nm[:2, face0] % face_size

        face1 = nmface[2] == 1
        nmface[:2, face1] = nm[:2, face1] % face_size

        face2 = nmface[2] == 2
        nmface[:2, face2] = nm[:2, face2] % face_size

        face3 = nmface[2] == 3
        nmface[0][face3] = face_size - nm[1][face3] - 1
        nmface[1][face3] = nm[0][face3] - face_size - 1

        face4 = nmface[2] == 4
        nmface[0][face4] = 2 * face_size - nm[1][face4] - 1
        nmface[1][face4] = nm[0][face4] - face_size - 1

        face5 = nmface[2] == 5
        nmface[0][face5] = 3 * face_size - nm[1][face5] - 1
        nmface[1][face5] = nm[0][face5] - face_size - 1

        return nmface.astype(int)

    @staticmethod
    def nmface2vuface(nmface: np.ndarray, proj_shape=None) -> np.ndarray:
        """

        :param proj_shape:
        :param nmface: (3, H, W)
        :return:
        """
        vuface = np.zeros(nmface.shape)

        if proj_shape is None:
            proj_shape = nmface.shape

        face_size = proj_shape[-1] / 3
        vuface[:2] = 2 * (nmface[:2, ...] + 0.5) / face_size - 1
        vuface[2] = nmface[2]
        return vuface

    @staticmethod
    def vuface2xyz_face(vuface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.zeros(vuface.shape)

        face0 = vuface[2] == 0
        xyz[0, face0] = -1
        xyz[1, face0] = vuface[0, face0]
        xyz[2, face0] = vuface[1, face0]

        face1 = vuface[2] == 1
        xyz[0, face1] = vuface[1, face1]
        xyz[1, face1] = vuface[0, face1]
        xyz[2, face1] = 1

        face2 = vuface[2] == 2
        xyz[0, face2] = 1
        xyz[1, face2] = vuface[0, face2]
        xyz[2, face2] = -vuface[1, face2]

        face3 = vuface[2] == 3
        xyz[0, face3] = -vuface[1, face3]
        xyz[1, face3] = 1
        xyz[2, face3] = vuface[0, face3]

        face4 = vuface[2] == 4
        xyz[0, face4] = -vuface[1, face4]
        xyz[1, face4] = vuface[0, face4]
        xyz[2, face4] = -1

        face5 = vuface[2] == 5
        xyz[0, face5] = -vuface[1, face5]
        xyz[1, face5] = -1
        xyz[2, face5] = -vuface[0, face5]
        face = vuface[2]

        return xyz, face

    @staticmethod
    def xyz2vuface(xyz: np.ndarray) -> np.ndarray:
        """

        :param xyz: (3, H, W)
        :return:
        """

        vuface = np.zeros(xyz.shape)
        abs_xyz = np.abs(xyz)

        def selection(v1, v2, v3, v4, v5):
            selection1 = np.logical_and(v1, v2)
            selection2 = np.logical_and(selection1, v3)
            selection3 = np.logical_and(selection2, v4)
            selection4 = np.logical_and(selection3, v5)
            return selection4

        face0 = selection(-xyz[0] >= -xyz[2], -xyz[0] > xyz[2], -xyz[0] >= -xyz[1], -xyz[0] > xyz[1], xyz[0] < 0)
        vuface[2][face0] = 0
        vuface[1][face0] = xyz[2][face0] / abs_xyz[0][face0]
        vuface[0][face0] = xyz[1][face0] / abs_xyz[0][face0]

        face1 = selection(xyz[2] >= -xyz[0], xyz[2] > xyz[0], xyz[2] >= -xyz[1], xyz[2] > xyz[1], xyz[2] > 0)
        vuface[2][face1] = 1
        vuface[1][face1] = xyz[0][face1] / abs_xyz[2][face1]
        vuface[0][face1] = xyz[1][face1] / abs_xyz[2][face1]

        face2 = selection(xyz[0] >= xyz[2], xyz[0] > -xyz[2], xyz[0] >= -xyz[1], xyz[0] > xyz[1], xyz[0] > 0)
        vuface[2][face2] = 2
        vuface[1][face2] = -xyz[2][face2] / abs_xyz[0][face2]
        vuface[0][face2] = xyz[1][face2] / abs_xyz[0][face2]

        face3 = selection(xyz[1] >= xyz[0], xyz[1] > -xyz[0], xyz[1] >= -xyz[2], xyz[1] > xyz[2], xyz[1] > 0)
        vuface[2][face3] = 3
        vuface[1][face3] = -xyz[0][face3] / abs_xyz[1][face3]
        vuface[0][face3] = xyz[2][face3] / abs_xyz[1][face3]

        face4 = selection(-xyz[2] >= xyz[0], -xyz[2] > -xyz[0], -xyz[2] >= -xyz[1], -xyz[2] > xyz[1], xyz[2] < 0)
        vuface[2][face4] = 4
        vuface[1][face4] = -xyz[0][face4] / abs_xyz[2][face4]
        vuface[0][face4] = xyz[1][face4] / abs_xyz[2][face4]

        face5 = selection(-xyz[1] >= xyz[0], -xyz[1] > -xyz[0], -xyz[1] >= xyz[2], -xyz[1] > -xyz[2], xyz[1] < 0)
        vuface[2][face5] = 5
        vuface[1][face5] = -xyz[0][face5] / abs_xyz[1][face5]
        vuface[0][face5] = -xyz[2][face5] / abs_xyz[1][face5]

        return vuface

    @staticmethod
    def vuface2nmface(vuface, proj_shape=None):
        """

        :param vuface:
        :param proj_shape: (h, w)
        :return:
        """
        nm_face = np.zeros(vuface.shape)
        nm_face[2] = vuface[2]

        if proj_shape is None:
            proj_shape = vuface.shape
        face_size = proj_shape[-1] / 3
        _face_size_2 = face_size / 2
        nm_face[:2] = np.round((vuface[:2, ...] + 1) * _face_size_2 - 0.5)
        return nm_face.astype(int)

    @staticmethod
    def nmface2cmp_face(nmface, proj_shape=None):
        """

        :param nmface:
        :param proj_shape: (h, w)
        :return:
        """
        new_shape = (2,) + nmface.shape[1:]
        nm = np.zeros(new_shape, dtype=int)

        if proj_shape is None:
            proj_shape = nmface.shape[-2:]
        face_size = proj_shape[-1] // 3

        face0 = nmface[2] == 0
        nm[0][face0] = nmface[0][face0]
        nm[1][face0] = nmface[1][face0]

        face1 = nmface[2] == 1
        nm[0][face1] = nmface[0][face1]
        nm[1][face1] = nmface[1][face1] + face_size

        face2 = nmface[2] == 2
        nm[0][face2] = nmface[0][face2]
        nm[1][face2] = nmface[1][face2] + 2 * face_size

        face3 = nmface[2] == 3
        nm[0][face3] = face_size + nmface[1][face3]
        nm[1][face3] = face_size - nmface[0][face3] - 1

        face4 = nmface[2] == 4
        nm[0][face4] = face_size + nmface[1][face4]
        nm[1][face4] = 2 * face_size - nmface[0][face4] - 1

        face5 = nmface[2] == 5
        nm[0][face5] = face_size + nmface[1][face5]
        nm[1][face5] = 3 * face_size - nmface[0][face5] - 1

        face = nmface[2]
        return nm, face


class CMP(Methods, ProjBase):
    def nm2xyz(self, nm: np.ndarray, proj_shape: Union[np.ndarray, tuple], rotate: bool = True):
        """
        CMP specific.

        :param nm: shape==(2,...)
        :param proj_shape: (N, M)
        :param rotate: True
        :return: x, y, z
        """
        nmface = self.cmp2nmface(nm, proj_shape)
        vuface = self.nmface2vuface(nmface, proj_shape)
        xyz, face = self.vuface2xyz_face(vuface)
        return xyz

    def xyz2nm(self, xyz: np.ndarray, proj_shape: np.ndarray = None, round_nm: bool = False, rotate=True):
        """
        CMP specific.

        :param rotate:
        :param xyz: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param proj_shape: the shape of projection that cover all sphere
        :param round_nm: round the coords? is not needed.
        :return:
        """
        vuface = self.xyz2vuface(xyz)
        nmface = self.vuface2nmface(vuface, proj_shape=proj_shape)
        cmp, face = self.nmface2cmp_face(nmface, proj_shape=proj_shape)

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
        # [ax.scatter(x, y, z, marker='o', color='red') for x, y, z in zip(xyz[0, 0:4140:100], xyz[1, 0:4140:100], xyz[2, 0:4140:100])]
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


def test_cmp():
    # cmp '144x96', '288x192','432x288','576x384'
    yaw_pitch_roll = np.deg2rad((70, 0, 0))
    height, width = 384, 576

    # Open Image
    frame_img: Union[Image, list] = Image.open('images/cmp1.png')
    frame_img = frame_img.resize((width, height))

    cmp = CMP(tiling='6x4', proj_res=f'{width}x{height}', fov='110x90')
    cmp.yaw_pitch_roll = yaw_pitch_roll
    compose(cmp, frame_img)


if __name__ == '__main__':
    test_cmp()
