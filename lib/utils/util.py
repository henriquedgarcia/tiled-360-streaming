import json
import os
import pickle
from collections import defaultdict
from collections.abc import Sequence, Hashable
from pathlib import Path
from subprocess import Popen, STDOUT, PIPE
from time import time
from typing import Union, Any, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from py360tools import ProjectionBase, ERP, CMP

from lib.assets.ansi_colors import Bcolors
from lib.assets.autodict import AutoDict


def run_command_os(command: str):
    """
    run with the shell
    :param command:
    :return:
    """
    print(command)
    os.system(command)


def show(img: np.ndarray):
    plt.imshow(img)
    plt.show()


def show1(projection: np.ndarray):
    plt.imshow(projection)
    plt.show()


def show2(projection: np.ndarray):
    frame_img = Image.fromarray(projection)
    frame_img.show()


def get_borders(*,
                coord_nm: Union[tuple, np.ndarray] = None,
                shape=None,
                thickness=1
                ):
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
    mat_x = np.array([[1, 0, 0],
                      [0, cos_rot[1], -sin_rot[1]],
                      [0, sin_rot[1], cos_rot[1]]])
    # yaw
    mat_y = np.array([[cos_rot[0], 0, sin_rot[0]],
                      [0, 1, 0],
                      [-sin_rot[0], 0, cos_rot[0]]])
    # roll
    mat_z = np.array([[cos_rot[2], -sin_rot[2], 0],
                      [sin_rot[2], cos_rot[2], 0],
                      [0, 0, 1]])

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
        # print(f'Frame {frame}, old={old:.3f}°, new={position:.3f}°, diff={diff :.3f}°')  # Want a log?

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


def find_keys(data: dict, level=0, result=None):
    """
    percorre um dicionário como uma árvore, recursivamente e anota quais são as keys em cada nível.
    """
    if result is None:
        result = defaultdict(set)

    for key, value in data.items():
        result[level].update([key])

        if isinstance(value, dict):
            find_keys(value, level + 1, result)
        else:
            continue

    result2 = {}
    for key in result:
        result2[key] = list(result[key])

    return result2


def percorrer_arvore_iterativo(dicionario):
    folhas = []
    pilha = [(dicionario, '')]

    while pilha:
        atual, caminho = pilha.pop()

        for chave, valor in atual.items():
            novo_caminho = f"{caminho}/{chave}" if caminho else chave

            if isinstance(valor, dict):
                pilha.append((valor, novo_caminho))
            else:
                folhas.append(f"{novo_caminho}: {valor}")

    return folhas


def draw_pixel_density():
    array = np.zeros([200, 200])
    n_array = np.zeros([199, 199])
    z = 1
    for i in range(200):
        x = (i - 100) / 200
        for j in range(200):
            y = (j - 100) / 200
            print(f'({i=}, {j}) => {x=}, {y=}, {z=} => {np.sqrt(x ** 2 + y ** 2 + z ** 2)} ')
            np.arccos(1 / (np.sqrt(x ** 2 + y ** 2 + 1)))
            array[j][i] = np.arccos(1 / (np.sqrt(x ** 2 + y ** 2 + 1)))
            if x <= 0 or y <= 0:
                continue
            n_array[j - 1][i - 1] = np.abs(array[j][i] - array[j - 1][i - 1])

    new_array = np.zeros([199 * 2, 199 * 3])

    for j in range(199 * 2):
        j_ = j % 199
        for i in range(199 * 3):
            i_ = i % 199
            new_array[j, i] = n_array[j_, i_] - 1

    plt.cla()
    plt.close()
    im = plt.imshow(new_array, cmap=plt.get_cmap('gray_r'))
    plt.axis('off')
    plt.colorbar(im, fraction=0.03)
    plt.title('Variação da densidade de pixels')
    plt.show()


def __geral__(): ...


def print_error(msg: str, end: str = '\n'):
    print(f'{Bcolors.RED}{msg}{Bcolors.ENDC}', end=end)


def save_json(data: Union[dict, list], filename: Union[str, Path], separators=(',', ':'), indent=None):
    filename = Path(filename)
    try:
        filename.write_text(json.dumps(data, separators=separators, indent=indent), encoding='utf-8')
    except (FileNotFoundError, OSError):
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(json.dumps(data, separators=separators, indent=indent), encoding='utf-8')


def load_json(filename: Union[str, Path], object_hook: type[dict] = None):
    filename = Path(filename)
    results = json.loads(filename.read_text(encoding='utf-8'), object_hook=object_hook)
    return results


def save_pickle(data: object, filename: Union[str, Path]):
    filename = Path(filename)
    try:
        filename.write_bytes(pickle.dumps(data, protocol=5))
    except FileNotFoundError:
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(pickle.dumps(data, protocol=5))


def load_pickle(filename: Path):
    filename = Path(filename)
    results = pickle.loads(filename.read_bytes())
    return results


def __coords__(): ...


def splitx(string: str) -> tuple[int, ...]:
    """
    Receive a string like "5x6x7" (no spaces) and return a tuple of ints, in
    this case, (5, 6, 7).
    :param string: A string of numbers separated with "x".
    :return: Return a list of int
    """
    return tuple(map(int, string.split('x')))


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


def lin_interpol(t: float, t_f: float, t_i: float, v_f: np.ndarray, v_i: np.ndarray) -> np.ndarray:
    m: np.ndarray = (v_f - v_i) / (t_f - t_i)
    v: np.ndarray = m * (t - t_i) + v_i
    return v


def make_tile_position_dict(video_shape, tiling_list):
    """

    :param video_shape:
    :param video_shape:
    :param tiling_list:
    :return:
    """
    proj_h, proj_w = video_shape
    resolution = f'{video_shape[1]}x{video_shape[0]}'
    tile_position_dict = AutoDict()

    for tiling in tiling_list:
        tiling_m, tiling_n = map(int, splitx(tiling))
        tile_w, tile_h = int(proj_w / tiling_m), int(proj_h / tiling_n)

        for tile in range(tiling_m * tiling_n):
            tile_x = tile % tiling_m
            tile_y = tile // tiling_m
            x1 = tile_x * tile_w  # not inclusive
            x2 = tile_x * tile_w + tile_w  # not inclusive
            y1 = tile_y * tile_h  # not inclusive
            y2 = tile_y * tile_h + tile_h  # not inclusive
            tile_position_dict[resolution][tiling][str(tile)] = [x1, y1, x2, y2]
    return tile_position_dict


def __misc__(): ...


def count_decoding(dectime_log: Path) -> int:
    """
    Count how many times the word "utime" appears in "log_file"
    :return:
    """
    times = len(get_times(dectime_log))
    return times


def get_times(filename: Path):
    content = filename.read_text(encoding='utf-8')
    times = []
    for line in content.splitlines():
        if 'utime' in line:
            t = float(line.strip().split(' ')[1].split('=')[1][:-1])
            if t > 0:
                times.append(t)
    return times


def decode_video(filename, threads=None, ui_prefix='', ui_suffix='\n'):
    """
    MakeDectime the filename HEVC video with "threads".
    :param filename:
    :param threads:
    :param ui_prefix:
    :param ui_suffix:
    :return:
    """

    cmd = make_decode_cmd(filename=filename, threads=threads)
    process, stdout = run_command(cmd, ui_prefix=ui_prefix, ui_suffix=ui_suffix)
    return stdout


def make_decode_cmd(filename, threads: int = None, codec: str = 'hevc', benchmark: bool = True):
    threads_opt = f'-threads {threads} ' if threads else ''
    benchmark = '-benchmark' if benchmark else ''
    codec = f'-codec {codec}'
    input_file = f'-i {filename.as_posix()}'

    cmd = (f'bash -c '
           f'"'
           f'bin/ffmpeg -hide_banner '
           f'{benchmark} '
           f'{codec} '
           f'{threads_opt}'
           f'{input_file} '
           f'-f null -'
           f'"')
    return cmd


def get_nested_value(data: dict[Hashable, Any], keys: Sequence[Hashable]) -> Any:
    """Fetch value from nested dict using a list of keys."""
    results = data
    for key in keys: results = results[key]
    return results


def set_nested_value(data: dict[Hashable, Any], keys: Sequence, value: Any):
    subtree = get_nested_value(data, keys[:-1])
    subtree[keys[-1]] = value


def run_command(cmd: str, folder: Optional[Path] = None, log_file: Optional[Path] = None, mode='w',
                ui_prefix='', ui_suffix='\n'):
    """

    :param cmd:
    :param folder:
    :param log_file:
    :param mode: like used by open()
    :param ui_prefix:
    :param ui_suffix:
    :return:
    """
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    process = Popen(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    stdout_lines = [cmd + '\n']
    print(ui_prefix, end='')
    while True:
        out = process.stdout.readline()
        if not out: break
        stdout_lines.append(out)
        print('.', end='')
    process.wait()
    stdout = ''.join(stdout_lines)
    print(' finish', end='')
    print(ui_suffix, end='')

    if log_file is not None:
        with open(log_file, mode) as f:
            f.write(stdout)

    return process, stdout


def __frame_handler__(): ...


def build_projection(proj_name, proj_res, tiling, vp_res, fov_res) -> ProjectionBase:
    if proj_name == 'erp':
        projection = ERP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res)
    elif proj_name == 'cmp':
        projection = CMP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res)
    else:
        raise TypeError(f'Unknown projection name: {proj_name}')
    return projection


def iter_video(video_path: Path, gray=True, dtype='float64'):
    cap = cv2.VideoCapture(f'{video_path}')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield frame.astype(dtype)


def get_frames(video_path, gray=True, dtype='float64'):
    cap = cv2.VideoCapture(f'{video_path}')
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(dtype))
    return frames


class LoadingUi:
    @staticmethod
    def start(prefix='\t'):
        print(f'{prefix}', end='')

    @staticmethod
    def increment():
        print('.', end='')

    @staticmethod
    def end(suffix='\n'):
        print(f'{suffix}', end='')
