import datetime
import json
from collections import defaultdict
from contextlib import contextmanager
from math import prod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import ndimage

from .transform import splitx
from .util import run_command, iter_frame


class AutoDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


class Bcolors:
    CYAN = '\033[96m'
    PINK = '\033[95m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    GRAY = '\033[90m'

    BG_GRAY = '\033[47m'
    BG_CYAN = '\033[46m'
    BG_PINK = '\033[45m'
    BG_BLUE = '\033[44m'
    BG_YELLOW = '\033[43m'
    BG_GREEN = '\033[42m'
    BG_RED = '\033[41m'
    BG_BLACK = '\033[40m'

    INVERT = '\033[6m'
    BLINK = '\033[5m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'
    WEAK = '\033[2m'
    BOLD = '\033[1m'

    CLEAR = '\033[A'

    ENDC = '\033[0m'


def print_fail(msg: str):
    print(f'{Bcolors.RED}{msg}{Bcolors.ENDC}')


class Config:
    config_data: dict

    def __init__(self, config_file: Union[Path, str]):
        print(f'Loading {config_file}.')

        self.config_file = Path(config_file)
        self.config_data = json.loads(self.config_file.read_text())

        videos_file = Path('config/' + self.config_data["videos_file"])
        videos_list = json.loads(videos_file.read_text())
        self.videos_list = videos_list['videos_list']

        for name in self.videos_list:
            self.videos_list[name].update({"fps": self.config_data['fps'],
                                           "gop": self.config_data['gop']})

        self.config_data['videos_list'] = self.videos_list

    def __getitem__(self, key):
        return self.config_data[key]

    def __setitem__(self, key, value):
        self.config_data[key] = value


class Factors:
    bins: Union[int, str] = None
    _video: str = None
    _name: str = None
    _proj: str = None
    quality_ref: str = '0'
    quality: str = None
    tiling: str = None
    metric: str = None
    tile: str = None
    chunk: str = None
    _name_list: list[str] = None
    _proj_list: list[str] = None
    config: Config
    user: int

    # <editor-fold desc="Main lists">
    @property
    def videos_list(self) -> dict[str, dict[str, Union[int, float, str]]]:
        return self.config.videos_list

    @property
    def name_list(self) -> list[str]:
        if self._name_list is None:
            _name_list = set([video.replace('_cmp', '').replace('_erp', '') for video in self.videos_list])
            self._name_list = sorted(list(_name_list))
        return self._name_list

    @property
    def proj_list(self) -> list[str]:
        if self._proj_list is None:
            _proj_set = set([self.videos_list[video]['projection'] for video in self.videos_list])
            self._proj_list = sorted(list(_proj_set))
        return self._proj_list

    @property
    def tiling_list(self) -> list[str]:
        return self.config['tiling_list']

    @property
    def quality_list(self) -> list[str]:
        quality_list = self.config['quality_list']
        return quality_list

    @property
    def tile_list(self) -> list[str]:
        splitx(self.tiling)
        n_tiles = prod(splitx(self.tiling))
        return list(map(str, range(n_tiles)))

    @property
    def chunk_list(self) -> list[str]:
        return list(map(str, range(1, int(self.duration) + 1)))

    @property
    def frame_list(self) -> list[str]:
        return list(range(int(self.duration * int(self.fps))))

    # </editor-fold>

    # <editor-fold desc="Video Property">
    @property
    def quality_str(self) -> str:
        return f'{self.rate_control}{self.quality}'

    @property
    def chunk_str(self) -> str:
        return f'chunk{self.chunk}'

    @property
    def tile_str(self) -> str:
        return f'tile{self.tile}'

    @property
    def group(self) -> str:
        return self.videos_list[self.video]['group']

    @property
    def video(self) -> str:
        if self._video is None and self._name is not None and self.proj is not None:
            return self._name.replace('_nas', f'_{self.proj}_nas')
        return self._video

    @video.setter
    def video(self, value):
        self._video = value

    @property
    def name(self) -> str:
        if self._name is None and self._video is not None:
            return self._video.replace('_cmp', '').replace('_erp', '')
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def proj(self) -> str:
        if self._proj is None:
            return self.vid_proj
        return self._proj

    @proj.setter
    def proj(self, value):
        self._proj = value

    @property
    def vid_proj(self) -> str:
        if self.video is None:
            return ''
        return self.videos_list[self.video]['projection']

    @property
    def scale(self) -> str:

        return self.videos_list[self.video]['scale']

    @property
    def resolution(self) -> str:
        return self.videos_list[self.video]['scale']

    @property
    def face_resolution(self) -> str:
        h, w, _ = self.face_shape
        return f'{w}x{h}'

    @property
    def face_shape(self) -> (int, int, int):
        h, w, _ = self.video_shape
        return round(h / 2), round(w / 3), 3

    @property
    def video_shape(self) -> tuple:
        w, h = splitx(self.videos_list[self.video]['scale'])
        return h, w, 3

    @property
    def fps(self) -> str:
        return self.config['fps']

    @property
    def rate_control(self) -> str:
        return self.config['rate_control']

    @property
    def gop(self) -> str:
        return self.config['gop']

    @property
    def duration(self) -> str:
        return self.videos_list[self.video]['duration']

    @property
    def offset(self) -> int:
        return int(self.videos_list[self.video]['offset'])

    @property
    def chunk_dur(self) -> int:
        return int(self.gop) // int(self.fps)

    @property
    def original(self) -> str:
        return self.videos_list[self.video]['original']

    # </editor-fold>

    # Tile Decoding Benchmark
    @property
    def decoding_num(self) -> int:
        return int(self.config['decoding_num'])

    # Metrics
    @property
    def metric_list(self) -> list[str]:
        return ['time', 'rate', 'time_std', 'MSE', 'WS-MSE', 'S-MSE']

    # GetTiles
    @property
    def fov(self) -> str:
        return self.config['fov']


class GlobalPaths(Factors):
    worker_name: str = None
    overwrite = False
    dectime_folder = Path('dectime')
    graphs_folder = Path('graphs')
    operation_folder = Path('')

    @property
    def project_path(self) -> Path:
        return Path('../results') / self.config['project']

    def tile_position(self):
        """
        Need video, tiling and tile
        :return: x1, x2, y1, y2
        """
        proj_h, proj_w = self.video_shape[:2]
        tiling_w, tiling_h = splitx(self.tiling)
        tile_w, tile_h = int(proj_w / tiling_w), int(proj_h / tiling_h)
        tile_m, tile_n = int(self.tile) % tiling_w, int(self.tile) // tiling_w
        x1 = tile_m * tile_w
        y1 = tile_n * tile_h
        x2 = tile_m * tile_w + tile_w  # not inclusive [...)
        y2 = tile_n * tile_h + tile_h  # not inclusive [...)
        return x1, y1, x2, y2


class Log(Factors):
    log_text: defaultdict

    @contextmanager
    def logger(self):
        self.log_text = defaultdict(list)

        try:
            yield
        finally:
            self.save_log()

    def log(self, error_code: str, filepath):
        self.log_text['video'].append(f'{self.video}')
        self.log_text['tiling'].append(f'{self.tiling}')
        self.log_text['quality'].append(f'{self.quality}')
        self.log_text['tile'].append(f'{self.tile}')
        self.log_text['chunk'].append(f'{self.chunk}')
        self.log_text['error'].append(error_code)
        self.log_text['parent'].append(f'{filepath.parent}')
        self.log_text['path'].append(f'{filepath.absolute()}')

    def save_log(self):
        cls_name = self.__class__.__name__
        filename = f'log/log_{cls_name}_{datetime.datetime.now()}.csv'
        filename = filename.replace(':', '-')
        df_log_text = pd.DataFrame(self.log_text)
        df_log_text.to_csv(filename, encoding='utf-8')


class Utils(GlobalPaths):
    command_pool: list

    def state_str(self):
        s = ''
        if self.proj:
            s += f'[{self.proj}]'
        if self.name:
            s += f'[{self.name}]'
        if self.tiling:
            s += f'[{self.tiling}]'
        if self.quality:
            s += f'[{self.rate_control}{self.quality}]'
        if self.tile:
            s += f'[tile{self.tile}]'
        if self.chunk:
            s += f'[chunk{self.chunk}]'
        return f'{self.__class__.__name__} {s}'

    @property
    def state(self):
        s = []
        if self.vid_proj:
            s.append(self.vid_proj)
        if self.name:
            s.append(self.name)
        if self.tiling:
            s.append(self.tiling)
        if self.quality:
            s.append(self.quality)
        if self.tile:
            s.append(self.tile)
        if self.chunk:
            s.append(self.chunk)
        return s

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.config.videos_list)} videos:\n'
              f'  operation: {self.__class__.__name__}\n'
              f'  project: {self.project_path}\n'
              f'  codec: {self.config["codec"]}\n'
              f'  fps: {self.config["fps"]}\n'
              f'  gop: {self.config["gop"]}\n'
              f'  qualities: {self.config["quality_list"]}\n'
              f'  patterns: {self.config["tiling_list"]}'
              )
        print('=' * 70)

    @contextmanager
    def multi(self):
        self.command_pool = []
        try:
            yield
            # with Pool(5) as p:
            #     p.map(run_command, self.command_pool)
            for command in self.command_pool:
                run_command(command)
        finally:
            pass


class SiTi:
    filename: Path
    previous_frame: Optional[np.ndarray]
    siti: dict

    def __init__(self, filename: Path):
        self.siti = defaultdict(list)
        self.previous_frame = None

        for n, frame in enumerate(iter_frame(filename)):
            si = self._calc_si(frame)
            self.siti['si'].append(si)

            ti = self._calc_ti(frame)
            self.siti['ti'].append(ti)

            print(f'\rSiTi - {filename.parts[-4:]}: frame={n}, si={si:.2f}, ti={ti:.3f}', end='')

        print('')

    def __getitem__(self, item) -> list:
        return self.siti[item]

    @staticmethod
    def _calc_si(frame: np.ndarray) -> (float, np.ndarray):
        """
        Calculate Spatial Information for a video frame. Calculate both vectors and so the magnitude.
        :param frame: A luma video frame in numpy ndarray format.
        :return: spatial information and sobel frame.
        """
        sob_y = ndimage.sobel(frame, axis=0)
        sob_x = ndimage.sobel(frame, axis=1, mode="wrap")
        sobel = np.hypot(sob_y, sob_x)
        si = sobel.std()
        return si

    def _calc_ti(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calculate Temporal Information for a video frame. If is a first frame,
        the information is zero.
        :param frame: A luma video frame in numpy ndarray format.
        :return: Temporal information and difference frame. If first frame the
        difference is zero array on same shape of frame.
        """
        try:
            difference = frame - self.previous_frame
        except TypeError:
            return 0.
        finally:
            self.previous_frame = frame
        return difference.std()
