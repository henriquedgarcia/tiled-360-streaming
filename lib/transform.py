from math import ceil
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ______basic_____(): ...


def cart2hcs(x, y, z) -> tuple[float, float]:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param float x: Coordinate from X axis
    :param float y: Coordinate from Y axis
    :param float z: Coordinate from Z axis
    :return: (azimuth, elevation) - in rad
    """
    # z-> x,
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(x, z)
    elevation = np.arcsin(-y / r)
    return azimuth, elevation


# todo: Verificar conscistência com arquivo quality.py
#
# def cart2hcs(x_y_z: np.ndarray) -> np.ndarray:
#     """
#     Convert from cartesian system to horizontal coordinate system in radians
#     :param x_y_z: 1D ndarray [x, y, z], or 2D array with shape=(N, 3)
#     :return: (azimuth, elevation) - in rad
#     """
#     r = np.sqrt(np.sum(x_y_z ** 2))
#     azimuth = np.arctan2(x_y_z[..., 0], x_y_z[..., 2])
#     elevation = np.arcsin(-x_y_z[..., 1] / r)
#     return np.array([azimuth, elevation]).T


def hcs2cart(azimuth: Union[np.ndarray, float], elevation: Union[np.ndarray, float]) \
        -> tuple[Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Convert from horizontal coordinate system  in radians to cartesian system.
    ISO/IEC JTC1/SC29/WG11/N17197l: Algorithm descriptions of projection format conversion and video quality metrics in 360Lib Version 5
    :param float elevation: Rad
    :param float azimuth: Rad
    :return: (x, y, z)
    """
    z = np.cos(elevation) * np.cos(azimuth)
    y = -np.sin(elevation)
    x = np.cos(elevation) * np.sin(azimuth)
    x, y, z = np.round([x, y, z], 6)
    return x, y, z


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
    azimuth, elevation = erp2hcs(n, m, shape)
    x, y, z = hcs2cart(azimuth, elevation)
    return np.array(x, y, z)


def cart2erp(x, y, z, shape):
    azimuth, elevation = cart2hcs(x, y, z)
    m, n = hcs2erp(azimuth, elevation, shape)
    return m, n


def erp2hcs(n: Union[np.ndarray, int], m: Union[np.ndarray, int], shape: Union[np.ndarray, tuple[int, int]]) -> Union[
    np.ndarray, tuple[float, float]]:
    """

    :param m: horizontal pixel coordinate
    :param n: vertical pixel coordinate
    :param shape: shape of projection in numpy format: (height, width)
    :return: (azimuth, elevation) - in rad
    """
    proj_h, proj_w = shape
    u = (m + 0.5) / proj_w
    v = (n + 0.5) / proj_h
    azimuth = (u - 0.5) * (2 * np.pi)
    elevation = (0.5 - v) * np.pi
    return azimuth, elevation


def hcs2erp(azimuth: float, elevation: float, shape: tuple) -> tuple[int, int]:
    """

    :param azimuth: in rad
    :param elevation: in rad
    :param shape: shape of projection in numpy format: (height, width)
    :return: (m, n) pixel coord using nearest neighbor
    """
    proj_h, proj_w = shape

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


def cart2uv_cmp(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    x, y, z = np.array(x), np.array(y), np.array(z)

    u = np.zeros(x.shape)
    v = np.zeros(x.shape)
    face = np.zeros(x.shape)

    ax, ay, az = np.abs((x, y, z))

    def selection(v1, v2, v3, v4, v5):
        selection1 = np.logical_and(v1, v2)
        selection2 = np.logical_and(selection1, v3)
        selection3 = np.logical_and(selection2, v4)
        selection4 = np.logical_and(selection3, v5)
        return selection4

    sel = selection(-x >= -z, -x > z, -x >= -y, -x > y, x < 0)
    face[sel] = 0
    u[sel] = z[sel]
    u[sel] = z[sel] / ax[sel]
    v[sel] = y[sel] / ax[sel]

    sel = selection(z >= -x, z > x, z >= -y, z > y, z > 0)
    face[sel] = 1
    u[sel] = x[sel] / az[sel]
    v[sel] = y[sel] / az[sel]

    sel = selection(x >= z, x > -z, x >= -y, x > y, x > 0)
    face[sel] = 2
    u[sel] = -z[sel] / ax[sel]
    v[sel] = y[sel] / ax[sel]

    sel = selection(y >= x, y > -x, y >= -z, y > z, y > 0)
    face[sel] = 3
    u[sel] = -x[sel] / ay[sel]
    v[sel] = z[sel] / ay[sel]

    sel = selection(-z >= x, -z > -x, -z >= -y, -z > y, z < 0)
    face[sel] = 4
    u[sel] = -x[sel] / az[sel]
    v[sel] = y[sel] / az[sel]

    sel = selection(-y >= x, -y > -x, -y >= z, -y > -z, y < 0)
    face[sel] = 5
    u[sel] = -x[sel] / ay[sel]
    v[sel] = z[sel] / ay[sel]

    return u, v, face


def uv_cmp2mn_face(u, v, side_size: int):
    m = int((u + 1) * side_size / 2)
    n = int((v + 1) * side_size / 2)
    return m, n


def mn_face2uv_cmp(m, n, side_size):
    u = (m + 0.5) * 2 / side_size - 1
    v = (n + 0.5) * 2 / side_size - 1
    return u, v


def uv_cmp2cart(u, v, face, side_size):
    x = np.zeros(shape=u.shape)
    y = np.zeros(shape=u.shape)
    z = np.zeros(shape=u.shape)

    # if face == 0:
    x[face == 0] = -1
    y[face == 0] = v[face == 0]
    z[face == 0] = u[face == 0]

    # elif face == 1:
    x[face == 1] = u[face == 1]
    y[face == 1] = v[face == 1]
    z[face == 1] = 1

    # elif face == 2:
    x[face == 2] = 1
    y[face == 2] = v[face == 2]
    z[face == 2] = -u[face == 2]
    # elif face == 3:
    x[face == 3] = -u[face == 3]
    y[face == 3] = 1
    z[face == 3] = v[face == 3]
    # elif face == 4:
    x[face == 4] = -u[face == 4]
    y[face == 4] = v[face == 4]
    z[face == 4] = -1
    # elif face == 5:
    x[face == 5] = -u[face == 5]
    y[face == 5] = -1
    z[face == 5] = -v[face == 5]

    return x, y, z


def mn_face2mn_proj(face_m, face_n, face, side_size):
    new_m, new_n = None, None
    if face == 0:
        new_m = face_m
        new_n = face_n
    elif face == 1:
        new_m = face_m + side_size
        new_n = face_n
    elif face == 2:
        new_m = face_m + 2 * side_size
        new_n = face_n
    elif face == 3:
        new_m = side_size - face_n - 1
        new_n = face_m + side_size
    elif face == 4:
        new_m = 2 * side_size - face_n - 1
        new_n = face_m + side_size
    elif face == 5:
        new_m = 3 * side_size - face_n - 1
        new_n = face_m + side_size

    return new_m, new_n


def cmp2mn_face(m, n, side_size):
    face_m = m % side_size
    face_n = n % side_size

    face = m // side_size + n // side_size * 3

    return face_m, face_n, face


def hcs2cmp(azimuth: float, elevation: float, shape: tuple) -> tuple[int, int, int]:
    """

    :param azimuth: in rad
    :param elevation: in rad
    :param shape: shape of projection in numpy format: (height, width)
    :return: (m, n, face) pixel coord using nearest neighbor
    """
    proj_h, proj_w = shape
    side_size = int(proj_h / 2)  # suppose the face is a square

    x, y, z = hcs2cart(azimuth, elevation)
    u, v, face = cart2uv_cmp(x, y, z)
    m, n = uv_cmp2mn_face(u, v, side_size)
    m, n = mn_face2mn_proj(m, n, face, side_size)
    return m, n, face


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


def ______test_____(): ...


def teste_hcs2cmp():
    shape = (400, 600)
    img = np.zeros(shape)
    for i in np.linspace(np.deg2rad(0), np.deg2rad(180), 100):
        yaw = i
        pitch = 0
        m, n, face = hcs2cmp(yaw, pitch, shape)
        img[n, m] = 255
        plt.scatter(m, n)
        print(f'{yaw=}, {pitch=}, {face=}')
        print(f'{m=}, {n=}, {face=}')

    # plt.imshow(img, cmap='gray')
    plt.xlim([0, 600])
    plt.ylim([0, 400])
    plt.show()
    plt.pause(0)


def teste_uv2mn_face():
    x = []
    y = []
    for coord in np.linspace(-1, 1, 30):
        m, n = uv_cmp2mn_face(coord, 0, 1080)
        x.append(m)
        y.append(n)
        plt.scatter(m, n)
    plt.show()
    plt.pause()


def teste_mn_face2mn_proj():
    for face in range(0, 1):
        for i in range(0, 1080, 60):
            m, n = mn_face2mn_proj(i, 0, face, 1080)
            plt.scatter(m, n)
            #
            m, n = mn_face2mn_proj(i, 1079, face, 1080)
            plt.scatter(m, n)
            #
            m, n = mn_face2mn_proj(0, i, face, 1080)
            plt.scatter(m, n)
            #
            m, n = mn_face2mn_proj(1079, i, face, 1080)
            plt.scatter(m, n)

    plt.show()


def teste_cart2uv_cmp():
    ax: plt.Axes
    ax = plt.figure().add_subplot()
    lu, lv, lface = [], [], []

    for angle in range(0, 360):
        x, y, z = hcs2cart(angle, np.deg2rad(0.0))
        u, v, face = cart2uv_cmp(x, y, z)
        lu.append(u)
        lv.append(v)
        lface.append(face)
        ax.scatter(u, v)

    plt.show()


def teste_hcs2cart():
    ax = plt.figure().add_subplot(projection='3d')
    lx, ly, lz = [], [], []

    for angle in range(300):
        elevation = np.deg2rad(angle)
        azimuth = np.deg2rad(angle)

        x, y, z = hcs2cart(azimuth, elevation)
        print(f'{x=}, {y=}, {z=}')
        lx.append(x)
        ly.append(-y)
        lz.append(z)

    lx, ly, lz = lx, lz, ly
    ax.plot(lx, ly, lz, label='parametric curve')
    ax.legend()
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)


if __name__ in '__main__':
    # teste_hcs2cart()
    # teste_hcs2uv()
    # teste_uv2mn_face()
    # teste_mn_face2mn_proj()
    teste_hcs2cmp()


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
