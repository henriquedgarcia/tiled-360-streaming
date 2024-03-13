from PIL import Image
import pickle
from math import ceil
from pathlib import Path
from time import time
from typing import Union

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg


def ______basic_____(): ...


def xyz2ea(xyz: np.ndarray) -> np.ndarray:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param xyz: shape = (3, ...)
    :return: np.ndarray([azimuth, elevation]) - in rad. shape = (2, ...)
    """
    new_shape = (2,) + xyz.shape[1:]
    ea = np.zeros(new_shape)
    # z-> x,
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
    newshape = (3,) + ae.shape[1:]
    xyz = np.zeros(newshape)
    xyz[0] = np.cos(ae[0]) * np.sin(ae[1])
    xyz[1] = -np.sin(ae[0])
    xyz[2] = np.cos(ae[0]) * np.cos(ae[1])
    xyz_r = np.round(xyz, 6)
    return xyz_r


def vp2cart(m, n, proj_shape, fov_shape):
    """
    Viewport generation with rectilinear projection

    :param m:
    :param n:
    :param proj_shape: (H, W)
    :param fov_shape: (fov_hor, fov_vert) in degree
    :return:
    """
    proj_h, proj_w = proj_shape
    fov_y, fov_x = map(np.deg2rad, fov_shape)
    half_fov_x, half_fov_y = fov_x / 2, fov_y / 2

    u = (m + 0.5) * 2 * np.tan(half_fov_x) / proj_w
    v = (n + 0.5) * 2 * np.tan(half_fov_y) / proj_h

    x = 1.
    y = -v + np.tan(half_fov_y)
    z = -u + np.tan(half_fov_x)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x = x / r
    y = y / r
    z = z / r

    return x, y, z


def ______erp_____(): ...


def nm2xyv(n_m_coord: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    ERP specific.

    :param n_m_coord: [(n, m], ...]
    :param shape: (H, W)
    :return:
    """
    v_u = (n_m_coord + (0.5, 0.5)) / shape
    elevation, azimuth = ((v_u - (0.5, 0.5)) * (-np.pi, 2 * np.pi)).T

    z = np.cos(elevation) * np.cos(azimuth)
    y = -np.sin(elevation)
    x = np.cos(elevation) * np.sin(azimuth)
    return np.array([x, y, z]).T


def erp2cart(n: Union[np.ndarray, int],
             m: Union[np.ndarray, int],
             shape: Union[np.ndarray, tuple[int, int]]) -> np.ndarray:
    """

    :param m: horizontal pixel coordinate
    :param n: vertical pixel coordinate
    :param shape: shape of projection in numpy format: (height, width)
    :return: x, y, z
    """
    ea = erp2hcs(n, m, shape)
    x, y, z = ea2xyz(ea)
    return np.array(x, y, z)


def cart2erp(x, y, z, shape):
    azimuth, elevation = xyz2ea(np.array([x, y, z]))
    m, n = hcs2erp(azimuth, elevation, shape)
    return m, n


def erp2hcs(n: Union[np.ndarray, int],
            m: Union[np.ndarray, int],
            proj_shape: Union[np.ndarray, tuple[int, int]]) \
        -> Union[np.ndarray, tuple[float, float]]:
    """

    :param m: horizontal pixel coordinate
    :param n: vertical pixel coordinate
    :param proj_shape: shape of projection in numpy format: (height, width)
    :return: (azimuth, elevation) - in rad
    """
    proj_h, proj_w = proj_shape
    u = (m + 0.5) / proj_w
    v = (n + 0.5) / proj_h
    azimuth = (u - 0.5) * (2 * np.pi)
    elevation = (0.5 - v) * np.pi
    return azimuth, elevation


def hcs2erp(azimuth: float, elevation: float, proj_shape: tuple) -> tuple[int, int]:
    """

    :param azimuth: angles in rad
    :param elevation: in rad
    :param proj_shape: shape of projection in numpy format: (height, width)
    :return: (m, n) pixel coord using nearest neighbor
    """
    proj_h, proj_w = proj_shape

    if azimuth >= np.pi or azimuth < -np.pi:
        azimuth = (azimuth + np.pi) % (2 * np.pi)
        azimuth = azimuth - np.pi

    if elevation > np.pi / 2:
        elevation = 2 * np.pi - elevation
    elif elevation < -np.pi / 2:
        elevation = -2 * np.pi - elevation

    u = azimuth / (2 * np.pi) + 0.5
    v = -elevation / np.pi + 0.5
    m = ceil(u * proj_w - 0.5)
    n = ceil(v * (proj_h - 1) - 0.5)
    return m, n


def ______cmp_____(): ...


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
    nmface[:2] = nm % face_size

    nmface_rotated = np.rot90(nmface, axes=(2, 1))
    nmface[:, nmface[2] == 3] = nmface_rotated[:, nmface_rotated[2] == 3]
    nmface[:, nmface[2] == 4] = nmface_rotated[:, nmface_rotated[2] == 4]
    nmface[:, nmface[2] == 5] = nmface_rotated[:, nmface_rotated[2] == 5]

    return nmface.astype(int)


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
    nmface_rotated = np.rot90(nmface, axes=(1, 2))
    nm[0][face3] = nmface_rotated[0][nmface_rotated[2] == 3] + face_size
    nm[1][face3] = nmface_rotated[1][nmface_rotated[2] == 3]

    face4 = nmface[2] == 4
    nm[0][face4] = nmface_rotated[0][nmface_rotated[2] == 4] + face_size
    nm[1][face4] = nmface_rotated[1][nmface_rotated[2] == 4] + face_size

    face5 = nmface[2] == 5
    nm[0][face5] = nmface_rotated[0][nmface_rotated[2] == 5] + face_size
    nm[1][face5] = nmface_rotated[1][nmface_rotated[2] == 5] + 2 * face_size
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

    normalize = np.vectorize(lambda m: 2 * (m + 0.5) / face_size - 1)
    vuface[:2] = normalize(nmface[:2, ...])
    vuface[2] = nmface[2]
    return vuface


def vuface2nmface(vuface, proj_shape=None):
    nm_face = np.zeros(vuface.shape)
    nm_face[2] = vuface[2]

    if proj_shape is None:
        proj_shape = vuface.shape
    face_size = proj_shape[-1] / 3
    _face_size_2 = face_size / 2

    denormalize = np.vectorize(lambda u: np.round((u + 1) * _face_size_2 - 0.5))
    nm_face[:2] = denormalize(vuface[:2, ...])
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

    face0 = selection(-xyz[0] >= -xyz[2],
                      -xyz[0] > xyz[2],
                      -xyz[0] >= -xyz[1],
                      -xyz[0] > xyz[1],
                      xyz[0] < 0)
    vuface[2][face0] = 0
    vuface[1][face0] = xyz[2][face0] / abs_xyz[0][face0]
    vuface[0][face0] = xyz[1][face0] / abs_xyz[0][face0]

    face1 = selection(xyz[2] >= -xyz[0],
                      xyz[2] > xyz[0],
                      xyz[2] >= -xyz[1],
                      xyz[2] > xyz[1],
                      xyz[2] > 0)
    vuface[2][face1] = 1
    vuface[1][face1] = xyz[0][face1] / abs_xyz[2][face1]
    vuface[0][face1] = xyz[1][face1] / abs_xyz[2][face1]

    face2 = selection(xyz[0] >= xyz[2],
                      xyz[0] > -xyz[2],
                      xyz[0] >= -xyz[1],
                      xyz[0] > xyz[1],
                      xyz[0] > 0)
    vuface[2][face2] = 2
    vuface[1][face2] = -xyz[2][face2] / abs_xyz[0][face2]
    vuface[0][face2] = xyz[1][face2] / abs_xyz[0][face2]

    face3 = selection(xyz[1] >= xyz[0],
                      xyz[1] > -xyz[0],
                      xyz[1] >= -xyz[2],
                      xyz[1] > xyz[2],
                      xyz[1] > 0)
    vuface[2][face3] = 3
    vuface[1][face3] = -xyz[0][face3] / abs_xyz[1][face3]
    vuface[0][face3] = xyz[2][face3] / abs_xyz[1][face3]

    face4 = selection(-xyz[2] >= xyz[0],
                      -xyz[2] > -xyz[0],
                      -xyz[2] >= -xyz[1],
                      -xyz[2] > xyz[1],
                      xyz[2] < 0)
    vuface[2][face4] = 4
    vuface[1][face4] = -xyz[0][face4] / abs_xyz[2][face4]
    vuface[0][face4] = xyz[1][face4] / abs_xyz[2][face4]

    face5 = selection(-xyz[1] >= xyz[0],
                      -xyz[1] > -xyz[0],
                      -xyz[1] >= xyz[2],
                      -xyz[1] > -xyz[2],
                      xyz[1] < 0)
    vuface[2][face5] = 5
    vuface[1][face5] = -xyz[0][face5] / abs_xyz[1][face5]
    vuface[0][face5] = -xyz[2][face5] / abs_xyz[1][face5]

    return vuface


def xyz2cmp_face(xyz: np.ndarray, proj_shape=None) -> tuple[np.ndarray, np.ndarray]:
    """

    :param proj_shape:
    :param xyz: shape(3, ...)
    :return: nm, face
    """
    vuface = xyz2vuface(xyz)
    nmface = vuface2nmface(vuface, proj_shape=proj_shape)
    nm, face = nmface2cmp_face(nmface, proj_shape=proj_shape)
    return nm, face


def cmp2xyz_face(nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    :type nm: np.ndarray
    :return: xyz, face
    """
    nmface = cmp2nmface(nm)
    vuface = nmface2vuface(nmface)
    xyz, face = vuface2xyz_face(vuface)
    return xyz, face


def ea2cmp_face(ea: np.ndarray, proj_shape: tuple = None) -> tuple[np.ndarray, np.ndarray]:
    """
    The face must be a square. proj_shape must have 3:2 ratio
    :param ea: in rad
    :param proj_shape: shape of projection in numpy format: (height, width)
    :return: (nm, face) pixel coord using nearest neighbor
    """
    if proj_shape is None:
        proj_shape = ea.shape

    xyz = ea2xyz(ea)
    nm, face = xyz2cmp_face(xyz, proj_shape=proj_shape)
    return nm, face


def cmp2ea_face(nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz, face = cmp2xyz_face(nm)
    ae = xyz2ea(xyz)
    return ae, face


def show_error(array):
    a = np.zeros(array.shape[1:])
    for j in range(a.shape[0]):
        for i in range(a.shape[1]):
            print(f"\r{j=}, {i=}      ", end='')
            for j2 in range(j, a.shape[0]):
                for i2 in range(i, a.shape[1]):
                    if j2 == j and i2 == i: continue
                    if (array[0, j, i] == array[0, j2, i2]
                            and array[1, j, i] == array[1, j2, i2]
                            and array[2, j, i] == array[2, j2, i2]):
                        a[j, i] = 255
                        print(f"\n\t{j2=}, {i2=} "
                              f"= vu({array[0, j2, i2]}, {array[1, j2, i2]}), face=({array[2, j, i]}, {array[2, j2, i2]})")
    return a


# def cmp2cart(n: int, m: int, shape: tuple[int, int], f: int = 0) -> tuple[float, float, float]:
#     x, y, z = None, None, None
#     proj_h, proj_w = shape
#     face_h = proj_h / 2  # face is a square. u
#     face_w = proj_w / 3  # face is a square. v
#     u = (m + 0.5) * 2 / face_h - 1
#     v = (n + 0.5) * 2 / face_w - 1
#     if f == 0:
#         x = 1.0
#         y = -v
#         z = -u
#     elif f == 1:
#         x = -1.0
#         y = -v
#         z = u
#     elif f == 2:
#         x = u
#         y = 1.0
#         z = v
#     elif f == 3:
#         x = u
#         y = -1.0
#         z = -v
#     elif f == 4:
#         x = u
#         y = -v
#         z = 1.0
#     elif f == 5:
#         x = -u
#         y = -v
#         z = -1.0
#     return x, y, z


def ______utils_____(): ...


def rot_matrix(yaw_pitch_roll: Union[np.ndarray, list]) -> np.ndarray:
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia. Use:
        X axis point to right
        Y axis point to down
        Z axis point to front

    Examples
    --------
    >> x, y, z = point
    >> mat = rot_matrix(yaw, pitch, roll)
    >> mat @ (x, y, z)

    :param yaw_pitch_roll: the rotation (yaw, pitch, roll) in rad.
    :return: A 3x3 matrix of rotation for (z,y,x) vector
    """
    cos_rot = np.cos(yaw_pitch_roll)
    sin_rot = np.sin(yaw_pitch_roll)

    # pitch
    mat_x = np.array(
        [[1, 0, 0],
         [0, cos_rot[1], -sin_rot[1]],
         [0, sin_rot[1], cos_rot[1]]]
    )
    # yaw
    mat_y = np.array(
        [[cos_rot[0], 0, sin_rot[0]],
         [0, 1, 0],
         [-sin_rot[0], 0, cos_rot[0]]]
    )
    # roll
    mat_z = np.array(
        [[cos_rot[2], -sin_rot[2], 0],
         [sin_rot[2], cos_rot[2], 0],
         [0, 0, 1]]
    )

    return mat_y @ mat_x @ mat_z


def check_deg(axis_name: str, value: float) -> float:
    """

    @param axis_name:
    @param value: in rad
    @return:
    """
    n_value = None
    if axis_name == 'azimuth':
        if value >= np.pi or value < -np.pi:
            n_value = (value + np.pi) % (2 * np.pi)
            n_value = n_value - np.pi
        return n_value
    elif axis_name == 'elevation':
        if value > np.pi / 2:
            n_value = 2 * np.pi - value
        elif value < -np.pi / 2:
            n_value = -2 * np.pi - value
        return n_value
    else:
        raise ValueError('"axis_name" not exist.')


def lin_interpol(t: float,
                 t_f: float, t_i: float,
                 v_f: np.ndarray, v_i: np.ndarray) -> np.ndarray:
    m: np.ndarray = (v_f - v_i) / (t_f - t_i)
    v: np.ndarray = m * (t - t_i) + v_i
    return v


def position2trajectory(positions_list, fps=30):
    # in rads: positions_list == (y, p, r)
    yaw_state = 0
    pitch_state = 0
    old_yaw = 0
    old_pitch = 0
    yaw_velocity = pd.Series(dtype=float)
    pitch_velocity = pd.Series(dtype=float)
    yaw_trajectory = []
    pitch_trajectory = []
    pi = np.pi

    for frame, position in enumerate(positions_list):
        """
        position: pd.Series
        position.index == []
        """
        yaw = position[0]
        pitch = position[1]

        if not frame == 1:
            yaw_diff = yaw - old_yaw
            if yaw_diff > 1.45:
                yaw_state -= 1
            elif yaw_diff < -200:
                yaw_state += 1

            pitch_diff = pitch - old_pitch
            if pitch_diff > 120:
                pitch_state -= 1
            elif pitch_diff < -120:
                pitch_state += 1
            # print(f'Frame {n}, old={old:.3f}°, new={position:.3f}°, diff={diff :.3f}°')  # Want a log?

        new_yaw = yaw + pi * yaw_state
        yaw_trajectory.append(new_yaw)

        new_pitch = pitch + pi / 2 * pitch_state
        pitch_trajectory.append(new_pitch)

        if frame == 1:
            yaw_velocity.loc[frame] = 0
            pitch_velocity.loc[frame] = 0
        else:
            yaw_velocity.loc[frame] = (yaw_trajectory[-1] - yaw_trajectory[-2]) * fps
            pitch_velocity.loc[frame] = (pitch_trajectory[-1] - pitch_trajectory[-2]) * fps

        old_yaw = yaw
        old_pitch = pitch

    # Filter
    padded_yaw_velocity = [yaw_velocity.iloc[0]] + list(yaw_velocity) + [yaw_velocity.iloc[-1]]
    yaw_velocity_filtered = [sum(padded_yaw_velocity[idx - 1:idx + 2]) / 3
                             for idx in range(1, len(padded_yaw_velocity) - 1)]

    padded_pitch_velocity = [pitch_velocity.iloc[0]] + list(pitch_velocity) + [pitch_velocity.iloc[-1]]
    pitch_velocity_filtered = [sum(padded_pitch_velocity[idx - 1:idx + 2]) / 3
                               for idx in range(1, len(padded_pitch_velocity) - 1)]

    # Scalar velocity
    yaw_speed = np.abs(yaw_velocity_filtered)
    pitch_speed = np.abs(pitch_velocity_filtered)

    # incomplete
    return yaw_speed, pitch_speed


def idx2xy(idx: int, shape: tuple):
    tile_x = idx % shape[1]
    tile_y = idx // shape[1]
    return tile_x, tile_y


def xy2idx(tile_x, tile_y, shape: tuple):
    idx = tile_x + tile_y * shape[0]
    return idx


def splitx(string: str) -> tuple[int, ...]:
    """
    Receive a string like "5x6x7" (no spaces) and return a tuple of ints, in
    this case, (5, 6, 7).
    :param string: A string of numbers separated with "x".
    :return: Return a list of int
    """
    return tuple(map(int, string.split('x')))


def mse2psnr(_mse: float) -> float:
    return 10 * np.log10((255. ** 2 / _mse))


def ______test_____(): ...


def test(func):
    print(f'Testing [{func.__name__}]: ', end='')
    start = time()
    try:
        func()
        print('OK.', end=' ')
    except AssertionError as e:
        print(f'{e.args[0]}', end=' ')
        pass
    final = time() - start
    print(f'Time = {final}')


class TestCMP:
    nm_test: np.ndarray
    nmface_test: np.ndarray
    vuface_test: np.ndarray
    xyz_face_test: tuple[np.ndarray, np.ndarray]
    ea_test: np.ndarray
    ae2cmp_test: np.ndarray
    ea_cmp_face_test: tuple[np.ndarray, np.ndarray]
    cmp2ea_test: np.ndarray

    def load_arrays(self):
        self.load_nm_file()
        self.load_nmface_file()
        self.load_vuface_file()
        self.load_xyz_file()
        self.load_ea_file()
        self.load_ea_cmp_file()

    def load_nm_file(self):
        nm_file = Path('data_test/nm.pickle')
        if nm_file.exists():
            self.nm_test = pickle.load(nm_file.open('rb'))
        else:
            shape = (200, 300)
            self.nm_test = np.mgrid[range(shape[0]), range(shape[1])]
            with open(nm_file, 'wb') as f:
                pickle.dump(self.nm_test, f)

    def load_nmface_file(self):
        nmface_file = Path('data_test/nmface.pickle')
        if nmface_file.exists():
            self.nmface_test = pickle.load(nmface_file.open('rb'))
        else:
            self.nmface_test = cmp2nmface(self.nm_test)
            with open(nmface_file, 'wb') as f:
                pickle.dump(self.nmface_test, f)

    def load_vuface_file(self):
        vuface_file = Path('data_test/vuface.pickle')
        if vuface_file.exists():
            self.vuface_test = pickle.load(vuface_file.open('rb'))
        else:
            self.vuface_test = nmface2vuface(self.nmface_test)
            with open(vuface_file, 'wb') as f:
                pickle.dump(self.vuface_test, f)

    def load_xyz_file(self):
        xyz_file = Path('data_test/xyz.pickle')
        if xyz_file.exists():
            self.xyz_face_test = pickle.load(xyz_file.open('rb'))
        else:
            self.xyz_face_test = vuface2xyz_face(self.vuface_test)
            with open(xyz_file, 'wb') as f:
                pickle.dump(self.xyz_face_test, f)

    def load_ea_file(self):
        ea_file = Path('data_test/ae.pickle')
        if ea_file.exists():
            self.ea_test = pickle.load(ea_file.open('rb'))
        else:
            self.ea_test, face1 = cmp2ea_face(self.nm_test)

            with open(ea_file, 'wb') as f:
                pickle.dump(self.ea_test, f)

    def load_ea_cmp_file(self):
        ea_cmp_file = Path('data_test/ea_cmp.pickle')

        if ea_cmp_file.exists():
            self.ea_cmp_face_test = pickle.load(ea_cmp_file.open('rb'))
        else:
            self.ea_cmp_face_test = ea2cmp_face(self.ea_test)

            with open(ea_cmp_file, 'wb') as f:
                pickle.dump(self.ea_cmp_face_test, f)

    def __init__(self):
        self.load_arrays()
        self.test()

    def test(self):
        test(self.teste_cmp2mn_face)
        test(self.teste_nmface2vuface)
        test(self.teste_vuface2xyz)

        test(self.teste_cmp2ea)

    def teste_cmp2mn_face(self):
        nmface = cmp2nmface(self.nm_test)
        nm, face = nmface2cmp_face(nmface)

        msg = ''
        if not np.array_equal(self.nm_test, nm):
            msg += 'Error in reversion'
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in nmface2cmp_face()'

        nm, face = nmface2cmp_face(self.nmface_test)
        if not np.array_equal(self.nm_test, nm):
            msg += 'Error in cmp2nmface()'

        assert msg == '', msg

    def teste_nmface2vuface(self):
        vuface = nmface2vuface(self.nmface_test)
        nmface = vuface2nmface(vuface)

        msg = ''
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in reversion'
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in nmface2vuface()'

        nmface = vuface2nmface(self.vuface_test)
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in vuface2nmface()'

        assert msg == '', msg

    def teste_vuface2xyz(self):
        xyz, face = vuface2xyz_face(self.vuface_test)
        vuface = xyz2vuface(xyz)

        msg = ''
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in reversion'
        if not np.array_equal(xyz, self.xyz_face_test[0]):
            msg += 'Error in vuface2xyz_face()'

        vuface = xyz2vuface(self.xyz_face_test[0])
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in xyz2vuface()'

        assert msg == '', msg

    # Test HCS
    def teste_cmp2ea(self):
        ea, face1 = cmp2ea_face(self.nm_test)
        nm, face2 = ea2cmp_face(ea)

        msg = ''
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in reversion'

        nm, face = ea2cmp_face(self.ea_test)
        if not np.array_equal(ea, self.ea_test):
            msg += 'Error in cmp2ea_face()'
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in ea2cmp_face()'

        assert msg == '', msg


def show_array(nm_array: np.ndarray, shape: tuple = None):
    """
          M
       +-->
       |
    n  v

    :param nm_array: shape (2, ...)
    :param shape: tuple (N, M)
    :return: None
    """
    if shape is None:
        shape = nm_array.shape[1:]
        if len(shape) < 2:
            shape = (np.max(nm_array[0]) + 1, np.max(nm_array[1]) + 1)
    array2 = np.zeros(shape, dtype=int)[nm_array[0], nm_array[1]] = 255
    Image.fromarray(array2).show()


if __name__ in '__main__':
    TestCMP()
