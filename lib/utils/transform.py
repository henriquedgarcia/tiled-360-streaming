# import pickle
# from pathlib import Path
# from typing import Union

# import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

# from lib.util import test


def ______basic_____(): ...


def xyz2ea(xyz: np.ndarray) -> np.ndarray:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param xyz: shape = (3, ...)
    :return: np.ndarray([azimuth, elevation]) - in rad. shape = (2, ...)
    """
    new_shape = (2,) + xyz.shape[1:]
    ea = np.zeros(new_shape)
    r = linalg.norm(xyz, axis=0)
    ea[0] = np.arcsin(-xyz[1] / r)
    ea[1] = np.arctan2(xyz[0], xyz[2])
    return ea


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


def vp2xyz(nm, proj_shape, fov_shape):
    """
    Viewport generation with rectilinear projection

    :param nm:
    :param proj_shape: (H, W)
    :param fov_shape: (fov_hor, fov_vert) in degree
    :return:
    """
    proj_h, proj_w = proj_shape
    fov_y, fov_x = map(np.deg2rad, fov_shape)
    half_fov_x, half_fov_y = fov_x / 2, fov_y / 2

    u = (nm[1] + 0.5) * 2 * np.tan(half_fov_x) / proj_w
    v = (nm[0] + 0.5) * 2 * np.tan(half_fov_y) / proj_h

    x = 1.
    y = -v + np.tan(half_fov_y)
    z = -u + np.tan(half_fov_x)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x = x / r
    y = y / r
    z = z / r

    return x, y, z


def ______erp_____(): ...


# def erp2vu(nm: np.ndarray, shape=None) -> np.ndarray:
#     if shape is None:
#         shape = nm.shape[1:]
#     vu = (nm + [[[0.5]], [[0.5]]]) / [[[shape[0]]], [[shape[1]]]]
#     return vu


# def vu2erp(vu, shape=None):
#     if shape is None:
#         shape = vu.shape[1:]
#
#     nm = vu * [[[shape[0]]], [[shape[1]]]]
#     nm = np.ceil(nm)
#     return nm


# def vu2ea(vu: np.ndarray) -> np.ndarray:
#     ea = (vu * [[[-np.pi]], [[2 * np.pi]]]) + [[[np.pi / 2]], [[-np.pi]]]
#     # ea = (vu - [[[0.5]], [[0.5]]]) * [[[-np.pi]], [[2 * np.pi]]]
#     return ea


def ea2vu(ea):
    vu = np.zeros(ea)
    vu[0] = -ea[0] / np.pi + 0.5
    vu[1] = ea[1] / (2 * np.pi) + 0.5
    return vu


# def erp2ea(nm: np.ndarray, shape=None) -> np.ndarray:
#     vu = erp2vu(nm, shape=shape)
#     ea = vu2ea(vu)
#     return ea


# def ea2erp(ea: np.ndarray, shape=None) -> np.ndarray:
#     """
#
#     :param ea: in rad
#     :param shape: shape of projection in numpy format: (height, width)
#     :return: (m, n) pixel coord using nearest neighbor
#     """
#     ea = normalize_ea(ea)
#     vu = ea2vu(ea)
#     nm = vu2erp(vu, shape)
#     return nm


# def erp2xyz(nm: np.ndarray, shape=None) -> np.ndarray:
#     """
#     ERP specific.
#
#     :param nm: [(n, m], ...]
#     :param shape: (H, W)
#     :return:
#     """
#     ea = erp2ea(nm, shape=shape)
#     xyz = ea2xyz(ea)
#     return xyz


# def xyz2erp(xyz, shape=None) -> np.ndarray:
#     ea = xyz2ea(xyz)
#     nm = ea2erp(ea, shape)
#     return nm


# class TestERP:
#     nm_test: Union[np.ndarray, list]
#     vu_test: np.ndarray
#     xyz_test: np.ndarray
#     ea_test: np.ndarray
#
#     def __init__(self):
#         self.load_arrays()
#         self.test()
#
#     def load_arrays(self):
#         self.load_nm_file()
#         self.load_vu_file()
#         self.load_ea_file()
#         self.load_xyz_file()
#
#     def load_nm_file(self):
#         nm_file = Path('data_test/ERP_nm.pickle')
#         if nm_file.exists():
#             self.nm_test = pickle.load(nm_file.open('rb'))
#         else:
#             shape = (200, 300)
#             self.nm_test = np.mgrid[0:shape[0], 0:shape[1]]
#             with open(nm_file, 'wb') as f:
#                 pickle.dump(self.nm_test, f)
#
#     def load_vu_file(self):
#         vu_file = Path('data_test/ERP_vu.pickle')
#         if vu_file.exists():
#             self.vu_test = pickle.load(vu_file.open('rb'))
#         else:
#             self.vu_test = erp2vu(self.nm_test)
#             with open(vu_file, 'wb') as f:
#                 pickle.dump(self.vu_test, f)
#
#     def load_ea_file(self):
#         ea_file = Path('data_test/ERP_ae.pickle')
#         if ea_file.exists():
#             self.ea_test = pickle.load(ea_file.open('rb'))
#         else:
#             self.ea_test, face1 = vu2ea(self.vu_test)
#
#             with open(ea_file, 'wb') as f:
#                 pickle.dump(self.ea_test, f)
#
#     def load_xyz_file(self):
#         xyz_file = Path('data_test/ERP_xyz.pickle')
#         if xyz_file.exists():
#             self.xyz_test = pickle.load(xyz_file.open('rb'))
#         else:
#             self.xyz_test = ea2xyz(self.ea_test)
#             with open(xyz_file, 'wb') as f:
#                 pickle.dump(self.xyz_test, f)
#
#     def test(self):
#         test(self.teste_erp2vu)
#         test(self.teste_vu2ea)
#
#     def teste_erp2vu(self):
#         vu = erp2vu(self.nm_test)
#         nm = vu2erp(vu)
#
#         msg = ''
#         if not np.array_equal(self.nm_test, nm):
#             msg += 'Error in reversion'
#         if not np.array_equal(vu, self.vu_test):
#             msg += 'Error in erp2vu()'
#
#         nm = vu2erp(self.vu_test)
#         if not np.array_equal(self.nm_test, nm):
#             msg += 'Error in vu2erp()'
#
#         assert msg == '', msg
#
#     def teste_vu2ea(self):
#         ea = vu2ea(self.vu_test)
#         vu = ea2vu(ea)
#
#         msg = ''
#         if not np.array_equal(vu, self.vu_test):
#             msg += 'Error in reversion'
#         if not np.array_equal(ea, self.ea_test):
#             msg += 'Error in vu2ea()'
#
#         vu = ea2vu(self.ea_test)
#         if not np.array_equal(vu, self.vu_test):
#             msg += 'Error in ea2vu()'
#
#         assert msg == '', msg
#
#     def teste_ea2xyz(self):
#         xyz = ea2xyz(self.ea_test)
#         ea = xyz2ea(xyz)
#
#         msg = ''
#         if not np.array_equal(ea, self.ea_test):
#             msg += 'Error in reversion'
#         if not np.array_equal(xyz, self.xyz_test):
#             msg += 'Error in ea2xyz()'
#
#         ea = xyz2ea(self.xyz_test)
#         if not np.array_equal(ea, self.ea_test):
#             msg += 'Error in xyz2ea()'
#
#         assert msg == '', msg


def ______cmp_____(): ...


# def cmp2nmface(nm: np.ndarray, proj_shape: tuple = None) -> np.ndarray:
#     """
#
#     :param proj_shape:
#     :param nm: shape(2, ...)
#                pixel coords in image; n = height, m = width
#     :return: nm_face(3, ...)
#     """
#     new_shape = (3,) + nm.shape[1:]
#     nmface = np.zeros(new_shape)
#
#     if proj_shape is None:
#         proj_shape = nm.shape
#
#     face_size = proj_shape[-1] // 3
#     nmface[2] = nm[1] // face_size + (nm[0] // face_size) * 3
#
#     face0 = nmface[2] == 0
#     nmface[:2, face0] = nm[:2, face0] % face_size
#
#     face1 = nmface[2] == 1
#     nmface[:2, face1] = nm[:2, face1] % face_size
#
#     face2 = nmface[2] == 2
#     nmface[:2, face2] = nm[:2, face2] % face_size
#
#     face3 = nmface[2] == 3
#     nmface[0][face3] = face_size - nm[1][face3] - 1
#     nmface[1][face3] = nm[0][face3] - face_size - 1
#
#     face4 = nmface[2] == 4
#     nmface[0][face4] = 2 * face_size - nm[1][face4] - 1
#     nmface[1][face4] = nm[0][face4] - face_size - 1
#
#     face5 = nmface[2] == 5
#     nmface[0][face5] = 3 * face_size - nm[1][face5] - 1
#     nmface[1][face5] = nm[0][face5] - face_size - 1
#
#     return nmface.astype(int)


def nmface2cmp_face(nmface, proj_shape=None):
    new_shape = (2,) + nmface.shape[1:]
    nm = np.zeros(new_shape, dtype=int)

    if proj_shape is None:
        proj_shape = nmface.shape
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


def vuface2nmface(vuface, proj_shape=None):
    nm_face = np.zeros(vuface.shape)
    nm_face[2] = vuface[2]

    if proj_shape is None:
        proj_shape = vuface.shape
    face_size = proj_shape[-1] / 3
    _face_size_2 = face_size / 2
    nm_face[:2] = np.round((vuface[:2, ...] + 1) * _face_size_2 - 0.5)
    return nm_face.astype(int)


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


# def xyz2cmp_face(xyz: np.ndarray, proj_shape=None) -> tuple[np.ndarray, np.ndarray]:
#     """
#
#     :param proj_shape:
#     :param xyz: shape(3, ...)
#     :return: nm, face
#     """
#     vuface = xyz2vuface(xyz)
#     nmface = vuface2nmface(vuface, proj_shape=proj_shape)
#     nm, face = nmface2cmp_face(nmface, proj_shape=proj_shape)
#     return nm, face


# def cmp2xyz_face(nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#
#     :type nm: np.ndarray
#     :return: xyz, face
#     """
#     nmface = cmp2nmface(nm)
#     vuface = nmface2vuface(nmface)
#     xyz, face = vuface2xyz_face(vuface)
#     return xyz, face


# def ea2cmp_face(ea: np.ndarray, proj_shape: tuple = None) -> tuple[np.ndarray, np.ndarray]:
#     """
#     The face must be a square. proj_shape must have 3:2 ratio
#     :param ea: in rad
#     :param proj_shape: shape of projection in numpy format: (height, width)
#     :return: (nm, face) pixel coord using nearest neighbor
#     """
#     if proj_shape is None:
#         proj_shape = ea.shape
#
#     xyz = ea2xyz(ea)
#     nm, face = xyz2cmp_face(xyz, proj_shape=proj_shape)
#     return nm, face


# def cmp2ea_face(nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     xyz, face = cmp2xyz_face(nm)
#     ae = xyz2ea(xyz)
#     return ae, face


# class TestCMP:
#     nm_test: Union[np.ndarray, list]
#     nmface_test: np.ndarray
#     vuface_test: np.ndarray
#     xyz_face_test: tuple[np.ndarray, np.ndarray]
#     ea_test: np.ndarray
#     ae2cmp_test: np.ndarray
#     ea_cmp_face_test: tuple[np.ndarray, np.ndarray]
#     cmp2ea_test: np.ndarray
#
#     def __init__(self):
#         self.load_arrays()
#         self.test()
#
#     def load_arrays(self):
#         self.load_nm_file()
#         self.load_nmface_file()
#         self.load_vuface_file()
#         self.load_xyz_file()
#         self.load_ea_file()
#         self.load_ea_cmp_file()
#
#     def test(self):
#         test(self.teste_cmp2mn_face)
#         test(self.teste_nmface2vuface)
#         test(self.teste_vuface2xyz)
#         test(self.teste_cmp2ea)
#
#     def teste_cmp2mn_face(self):
#         nmface = cmp2nmface(self.nm_test)
#         nm, face = nmface2cmp_face(nmface)
#
#         msg = ''
#         if not np.array_equal(self.nm_test, nm):
#             msg += 'Error in reversion'
#         if not np.array_equal(nmface, self.nmface_test):
#             msg += 'Error in nmface2cmp_face()'
#
#         nm, face = nmface2cmp_face(self.nmface_test)
#         if not np.array_equal(self.nm_test, nm):
#             msg += 'Error in cmp2nmface()'
#
#         assert msg == '', msg
#
#     def teste_nmface2vuface(self):
#         vuface = nmface2vuface(self.nmface_test)
#         nmface = vuface2nmface(vuface)
#
#         msg = ''
#         if not np.array_equal(nmface, self.nmface_test):
#             msg += 'Error in reversion'
#         if not np.array_equal(vuface, self.vuface_test):
#             msg += 'Error in nmface2vuface()'
#
#         nmface = vuface2nmface(self.vuface_test)
#         if not np.array_equal(nmface, self.nmface_test):
#             msg += 'Error in vuface2nmface()'
#
#         assert msg == '', msg
#
#     def teste_vuface2xyz(self):
#         xyz, face = vuface2xyz_face(self.vuface_test)
#         vuface = xyz2vuface(xyz)
#
#         msg = ''
#         if not np.array_equal(vuface, self.vuface_test):
#             msg += 'Error in reversion'
#         if not np.array_equal(xyz, self.xyz_face_test[0]):
#             msg += 'Error in vuface2xyz_face()'
#
#         vuface = xyz2vuface(self.xyz_face_test[0])
#         if not np.array_equal(vuface, self.vuface_test):
#             msg += 'Error in xyz2vuface()'
#
#         assert msg == '', msg
#
#     def teste_cmp2ea(self):
#         ea, face1 = cmp2ea_face(self.nm_test)
#         nm, face2 = ea2cmp_face(ea)
#
#         msg = ''
#         if not np.array_equal(nm, self.nm_test):
#             msg += 'Error in reversion'
#
#         nm, face = ea2cmp_face(self.ea_test)
#         if not np.array_equal(ea, self.ea_test):
#             msg += 'Error in cmp2ea_face()'
#         if not np.array_equal(nm, self.nm_test):
#             msg += 'Error in ea2cmp_face()'
#
#         assert msg == '', msg
#
#     def load_nm_file(self):
#         nm_file = Path('data_test/nm.pickle')
#         if nm_file.exists():
#             self.nm_test = pickle.load(nm_file.open('rb'))
#         else:
#             shape = (200, 300)
#             self.nm_test = np.mgrid[0:shape[0], 0:shape[1]]
#             with open(nm_file, 'wb') as f:
#                 pickle.dump(self.nm_test, f)
#
#     def load_nmface_file(self):
#         nmface_file = Path('data_test/nmface.pickle')
#         if nmface_file.exists():
#             self.nmface_test = pickle.load(nmface_file.open('rb'))
#         else:
#             self.nmface_test = cmp2nmface(self.nm_test)
#             with open(nmface_file, 'wb') as f:
#                 pickle.dump(self.nmface_test, f)
#
#     def load_vuface_file(self):
#         vuface_file = Path('data_test/vuface.pickle')
#         if vuface_file.exists():
#             self.vuface_test = pickle.load(vuface_file.open('rb'))
#         else:
#             self.vuface_test = nmface2vuface(self.nmface_test)
#             with open(vuface_file, 'wb') as f:
#                 pickle.dump(self.vuface_test, f)
#
#     def load_xyz_file(self):
#         xyz_file = Path('data_test/xyz.pickle')
#         if xyz_file.exists():
#             self.xyz_face_test = pickle.load(xyz_file.open('rb'))
#         else:
#             self.xyz_face_test = vuface2xyz_face(self.vuface_test)
#             with open(xyz_file, 'wb') as f:
#                 pickle.dump(self.xyz_face_test, f)
#
#     def load_ea_file(self):
#         ea_file = Path('data_test/ae.pickle')
#         if ea_file.exists():
#             self.ea_test = pickle.load(ea_file.open('rb'))
#         else:
#             self.ea_test, face1 = cmp2ea_face(self.nm_test)
#
#             with open(ea_file, 'wb') as f:
#                 pickle.dump(self.ea_test, f)
#
#     def load_ea_cmp_file(self):
#         ea_cmp_file = Path('data_test/ea_cmp.pickle')
#
#         if ea_cmp_file.exists():
#             self.ea_cmp_face_test = pickle.load(ea_cmp_file.open('rb'))
#         else:
#             self.ea_cmp_face_test = ea2cmp_face(self.ea_test)
#
#             with open(ea_cmp_file, 'wb') as f:
#                 pickle.dump(self.ea_cmp_face_test, f)


if __name__ in '__main__':
    # TestCMP()
    # TestERP()
    ...