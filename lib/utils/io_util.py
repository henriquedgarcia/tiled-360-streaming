import json
import pickle
from pathlib import Path
from typing import Union, cast, Literal

import pandas as pd

from lib.assets.ansi_colors import Bcolors


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


def save_hdf(data: pd.DataFrame, filename: Union[str, Path], key='default',
             mode=cast(Literal["a", "w", "r+"], 'w'), complevel=9):
    filename = Path(filename)

    try:
        data.to_hdf(filename, key=key, mode=mode, complevel=complevel)
    except (FileNotFoundError, OSError):
        filename.parent.mkdir(parents=True, exist_ok=True)
        data.to_hdf(filename, key=key, mode=mode, complevel=complevel)


def load_hdf(filename: Union[str, Path]) -> pd.DataFrame:
    filename = Path(filename)
    results: pd.DataFrame = pd.read_hdf(filename)
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
