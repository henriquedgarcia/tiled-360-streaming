
import json
import os
import pickle
from pathlib import Path
from subprocess import run, STDOUT, PIPE
from time import time
from typing import Union

import pandas as pd
from PIL import Image

import numpy as np
import skvideo.io
from matplotlib import pyplot as plt


def save_json(data: Union[dict, list], filename: Union[str, Path], separators=(',', ':'), indent=None):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=separators, indent=indent)


def load_json(filename, object_hook=None):
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f, object_hook=object_hook)
    return results


def save_pickle(data: object, filename: Union[str, Path]):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=5)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def count_decoding(dectime_log: Path) -> int:
    """
    Count how many times the word "utime" appears in "log_file"
    :return:
    """
    try:
        content = dectime_log.read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        print('ERROR: UnicodeDecodeError. Cleaning.')
        dectime_log.unlink()
        return 0
    except FileNotFoundError:
        print('ERROR: FileNotFoundError. Return 0.')
        return 0

    return len(['' for line in content if 'utime' in line])


def decode_file(filename, threads=None):
    """
    Decode the filename HEVC video with "threads".
    :param filename:
    :param threads:
    :return:
    """
    cmd = (f'bin/ffmpeg -hide_banner -benchmark '
           f'-codec hevc '
           f'{"" if not threads else f"-threads {threads} "}'
           f'-i {filename.as_posix()} '
           f'-f null -')
    if os.name == 'nt':
        cmd = f'bash -c "{cmd}"'

    process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    return process.stdout


def run_command(command: str):
    """
    run with the shell
    :param command:
    :return:
    """
    print(command)
    os.system(command)


def get_times(content: str):
    times = []
    for line in content.splitlines():
        if 'utime' in line:
            t = float(line.strip().split(' ')[1].split('=')[1][:-1])
            if t > 0:
                times.append(t)
    return times


def show(img: np.ndarray):
    plt.imshow(img)
    plt.show()


def iter_frame(video_path, gray=True, dtype='float64'):
    vreader = skvideo.io.vreader(f'{video_path}', as_grey=gray)
    # frames = []
    for frame in vreader:
        if gray:
            _, height, width, _ = frame.shape
            frame = frame.reshape((height, width)).astype(dtype)
        # frames.append(frame)
        yield frame
    # return frames


def show1(projection: np.ndarray):
    plt.imshow(projection)
    plt.show()


def show2(projection: np.ndarray):
    frame_img = Image.fromarray(projection)
    frame_img.show()


def get_borders(*,
                coord_nm: Union[tuple, np.ndarray] = None,
                shape=None,
                thickness=1):
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
    mat_x = np.array([[1, 0, 0], [0, cos_rot[1], -sin_rot[1]], [0, sin_rot[1], cos_rot[1]]])
    # yaw
    mat_y = np.array([[cos_rot[0], 0, sin_rot[0]], [0, 1, 0], [-sin_rot[0], 0, cos_rot[0]]])
    # roll
    mat_z = np.array([[cos_rot[2], -sin_rot[2], 0], [sin_rot[2], cos_rot[2], 0], [0, 0, 1]])

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


def lin_interpol(t: float, t_f: float, t_i: float, v_f: np.ndarray, v_i: np.ndarray) -> np.ndarray:
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
                pitch_state += 1  # print(f'Frame {n}, old={old:.3f}°, new={position:.3f}°, diff={diff :.3f}°')  # Want a log?

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
    yaw_velocity_filtered = [sum(padded_yaw_velocity[idx - 1:idx + 2]) / 3 for idx in range(1, len(padded_yaw_velocity) - 1)]

    padded_pitch_velocity = [pitch_velocity.iloc[0]] + list(pitch_velocity) + [pitch_velocity.iloc[-1]]
    pitch_velocity_filtered = [sum(padded_pitch_velocity[idx - 1:idx + 2]) / 3 for idx in range(1, len(padded_pitch_velocity) - 1)]

    # Scalar velocity
    yaw_speed = np.abs(yaw_velocity_filtered)
    pitch_speed = np.abs(pitch_velocity_filtered)

    # incomplete
    return yaw_speed, pitch_speed


def idx2xy(idx: Union[int, str], shape: tuple):
    """

    :param idx: index
    :param shape: (height, width)
    :return: tuple
    """
    idx = int(idx)
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
