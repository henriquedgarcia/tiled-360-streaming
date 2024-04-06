import json
import os
import pickle
from pathlib import Path
from subprocess import run, STDOUT, PIPE, Popen
from typing import Union

import numpy as np
import skvideo.io
from matplotlib import pyplot as plt


def save_json(data: Union[dict, list], filename: Union[str, Path], separators=(',', ':'), indent=None):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=separators, indent=indent)


def load_json(filename, object_hook=dict):
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
