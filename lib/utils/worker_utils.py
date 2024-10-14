import json
import os
import pickle
from functools import reduce
from pathlib import Path
from subprocess import STDOUT, PIPE, Popen
from typing import Union

import numpy as np
import skvideo.io

from lib.assets.ansi_colors import Bcolors
from lib.assets.autodict import AutoDict


def __geral__(): ...


def print_error(msg: str, end: str = '\n'):
    print(f'{Bcolors.RED}{msg}{Bcolors.ENDC}',
          end=end)


def load_json(filename, object_hook=None):
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f, object_hook=object_hook)
    return results


def save_json(data: Union[dict, list], filename: Union[str, Path], separators=(',', ':'), indent=None):
    try:
        json_dump(data, filename, separators, indent)
    except FileNotFoundError:
        filename.parent.mkdir(parents=True, exist_ok=True)
        json_dump(data, filename, separators, indent)


def json_dump(data, filename, separators=(',', ':'), indent=None):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=separators, indent=indent)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def save_pickle(data: object, filename: Union[str, Path]):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=5)


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
    try:
        times = len(get_times(dectime_log, only_count=True))
    except FileNotFoundError:
        print('ERROR: FileNotFoundError. Return 0.')
        times = 0
    return times


def get_times(filename: Path, only_count=False):
    content = filename.read_text(encoding='utf-8')
    times = []
    for line in content.splitlines():
        if 'utime' in line:
            if only_count:
                times.append('')
                continue
            t = float(line.strip().split(' ')[1].split('=')[1][:-1])
            if t > 0:
                times.append(t)
    return times


def decode_video(filename, threads=None, ui_prefix='', ui_suffix='\n'):
    """
    Decode the filename HEVC video with "threads".
    :param filename:
    :param threads:
    :param ui_prefix:
    :param ui_suffix:
    :return:
    """
    cmd = make_decode_cmd(filename=filename, threads=threads)
    process, stdout = run_command(cmd, ui_prefix=ui_prefix, ui_suffix=ui_suffix)
    return stdout


def make_decode_cmd(filename, threads: int = None):
    threads_opt = f'-threads {threads} ' if threads else ''

    cmd = (f'bash -c '
           f'"'
           f'bin/ffmpeg -hide_banner -benchmark '
           f'-codec hevc '
           f'{threads_opt}'
           f'-i {filename.as_posix()} '
           f'-f null -'
           f'"')
    return cmd


def get_nested_value(data, keys):
    """Fetch value from nested dict using a list of keys."""
    try:
        return reduce(lambda d, key: d[key], keys, data)
    except KeyError as e:
        raise KeyError(f"Key not found: {e}")
    except TypeError as e:
        raise TypeError(f"Invalid structure: {e}")


def run_command(cmd, folder=None, log_file=None, mode='w', ui_prefix='', ui_suffix='\n'):
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

    ui = LoadingUi()
    ui.start(prefix=ui_prefix)
    process = Popen(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    stdout_lines = [cmd + '\n']

    while True:
        ui.increment()
        out = process.stdout.readline()
        if not out:
            break
        stdout_lines.append(out)
    ui.end(suffix=ui_suffix)

    if log_file is not None:
        with open(log_file, mode) as f:
            f.writelines(stdout_lines)
    stdout = ''.join(stdout_lines)

    return process, stdout


def __frame_handler__(): ...


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
