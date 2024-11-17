import json
import pickle
from functools import reduce
from pathlib import Path
from subprocess import STDOUT, PIPE, Popen
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

from lib.assets.ansi_colors import Bcolors
from lib.assets.autodict import AutoDict


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
    times = len(get_times(dectime_log, only_count=True))
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


def get_nested_value(data, keys):
    """Fetch value from nested dict using a list of keys."""
    try:
        return reduce(lambda d, key: d[key], keys, data)
    except KeyError as e:
        raise KeyError(f"Key not found: {e}")
    except TypeError as e:
        raise TypeError(f"Invalid structure: {e}")


def run_command(cmd: str, folder: Path = None, log_file: Path = None, mode='w',
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

    with tqdm(desc=f'{ui_prefix}Running Command', total=float("inf")) as bar:
        process = Popen(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
        stdout_lines = [cmd + '\n']
        while True:
            out = process.stdout.readline()
            if not out: break
            stdout_lines.append(out)
            bar.update(len(stdout_lines))
        process.wait()
        stdout = ''.join(stdout_lines)
        print(ui_suffix, end='')

    if log_file is not None:
        with open(log_file, mode) as f:
            f.write(stdout)

    return process, stdout


def __frame_handler__(): ...


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
