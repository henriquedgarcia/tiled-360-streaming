import datetime
import json
from collections import defaultdict
from contextlib import contextmanager
from math import prod
from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import Optional, Union

import pandas as pd

from .transform import splitx
from .util import run_command


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


def print_error(msg: str, end: str = '\n'):
    print(f'{Bcolors.RED}{msg}{Bcolors.ENDC}', end=end)


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
            self.videos_list[name].update({"fps": self.config_data['fps'], "gop": self.config_data['gop']})

        self.config_data['videos_list'] = self.videos_list

    def __getitem__(self, key):
        return self.config_data[key]

    def __setitem__(self, key, value):
        self.config_data[key] = value


class Factors:
    _video: Optional[str] = None
    _name: Optional[str] = None
    _proj: Optional[str] = None
    _quality: Optional[str] = None
    _tiling: Optional[str] = None
    _tile: Optional[str] = None
    _chunk: Optional[str] = None
    _metric: Optional[str] = None
    bins: Optional[Union[int, str]] = None
    quality_ref: str = '0'
    _name_list: Optional[list[str]] = None
    _proj_list: Optional[list[str]] = None
    config: Config
    user: int
    turn: int

    # <editor-fold desc="Config">
    def config_props(self):
        ...

    @property
    def project(self):
        return self.config['project']

    @property
    def dataset_name(self):
        return self.config['dataset_name']

    @property
    def error_metric(self):
        return self.config['error_metric']

    @property
    def decoding_num(self) -> int:
        return int(self.config['decoding_num'])

    @property
    def fov(self) -> str:
        return self.config['fov']

    @property
    def codec(self) -> str:
        return self.config['codec']

    @property
    def fps(self) -> str:
        return self.config['fps']

    @property
    def gop(self) -> str:
        return self.config['gop']

    @property
    def rate_control(self) -> str:
        return self.config['rate_control']

    @property
    def distributions(self) -> str:
        return self.config['distributions']

    @property
    def original_quality(self) -> str:
        return self.config['original_quality']

    # </editor-fold>

    # <editor-fold desc="Main lists">
    def main_list_props(self):
        ...

    _metric_list = ['time', 'time_std', 'rate', 'SSIM', 'MSE', 'WS-MSE', 'S-MSE']

    @property
    def metric_list(self) -> list[str]:
        return self._metric_list

    @property
    def video_list(self) -> dict[str, dict[str, Union[int, float, str]]]:
        return self.config.videos_list

    @property
    def name_list(self) -> list[str]:
        if self._name_list is None:
            _name_list = set([video.replace('_cmp', '').replace('_erp', '') for video in self.video_list])
            self._name_list = sorted(list(_name_list))
        return self._name_list

    @property
    def proj_list(self) -> list[str]:
        if self._proj_list is None:
            _proj_set = set([self.video_list[video]['projection'] for video in self.video_list])
            self._proj_list = sorted(list(_proj_set))
        return self._proj_list

    @property
    def tiling_list(self) -> list[str]:
        return self.config['tiling_list']

    @property
    def quality_list(self) -> list[str]:
        return self.config['quality_list']

    @property
    def tile_list(self) -> list[str]:
        splitx(self.tiling)
        n_tiles = prod(splitx(self.tiling))
        for tile in range(n_tiles):
            yield str(tile)

    @property
    def chunk_list(self) -> list[str]:
        for chunk in range(1, int(self.duration) + 1):
            yield str(chunk)

    @property
    def frame_list(self) -> list[str]:
        for frame in range(self.n_frames):
            yield str(frame)

    # </editor-fold>

    # <editor-fold desc="state strings">
    def state_strings_props(self):
        ...

    @property
    def quality_str(self) -> str:
        return f'{self.rate_control}{self.quality}'

    @property
    def chunk_str(self) -> str:
        return f'chunk{self.chunk}'

    @property
    def tile_str(self) -> str:
        return f'tile{self.tile}'

    # </editor-fold>

    # <editor-fold desc="Video Property">
    def video_props(self):
        ...

    @property
    def original(self) -> str:
        return self.video_list[self.video]['original']

    @property
    def scale(self) -> str:
        return self.video_list[self.video]['scale']

    @property
    def projection(self) -> str:
        return self.video_list[self.video]['projection']

    @property
    def offset(self) -> int:
        return int(self.video_list[self.video]['offset'])

    @property
    def duration(self) -> str:
        return '60'

    @property
    def group(self) -> str:
        return self.video_list[self.video]['group']

    @property
    def resolution(self) -> str:
        return self.scale

    @property
    def n_frames(self) -> int:
        return int(self.duration) * int(self.fps)

    @property
    def chunk_dur(self) -> int:
        return int(self.gop) // int(self.fps)

    @property
    def video_shape(self) -> tuple:
        w, h = splitx(self.resolution)
        return h, w, 3

    @property
    def video_h(self) -> tuple:
        return self.video_shape[0]

    @property
    def video_w(self) -> tuple:
        return self.video_shape[1]

    # </editor-fold>

    # <editor-fold desc="State">
    def states_props(self):
        ...

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value

    @property
    def video(self) -> str:
        if self._video is None and None not in (self._name, self._proj):
            return self._name.replace('_nas', f'_{self._proj}_nas')
        return self._video

    @video.setter
    def video(self, value):
        self._video = value
        self._name = None
        self._proj = None

    @property
    def name(self) -> str:
        if self._name is None and self._video is not None:
            return self._video.replace('_cmp', '').replace('_erp', '')
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self._video = None

    @property
    def proj(self) -> str:
        if self._proj is None and self._video is not None:
            return self.projection
        return self._proj

    @proj.setter
    def proj(self, value):
        self._video = None
        self._proj = value

    @property
    def quality(self) -> str:
        return self._quality

    @quality.setter
    def quality(self, value):
        self._quality = value

    @property
    def tiling(self) -> str:
        return self._tiling

    @tiling.setter
    def tiling(self, value):
        self._tiling = value

    @property
    def tile(self) -> str:
        return self._tile

    @tile.setter
    def tile(self, value):
        self._tile = value

    @property
    def chunk(self) -> str:
        return self._chunk

    @chunk.setter
    def chunk(self, value):
        self._chunk = value

    # </editor-fold>

    # <editor-fold desc="Others">
    def others(self):
        ...

    @property
    def cmp_face_resolution(self) -> str:
        h, w, _ = self.cmp_face_shape
        return f'{w}x{h}'

    @property
    def cmp_face_shape(self) -> (int, int, int):
        h, w, c = self.video_shape
        return round(h / 2), round(w / 3), c  # </editor-fold>


class ContextObj(Factors):
    @contextmanager
    def context_metric(self):
        for self.metric in self.metric_list:
            yield

    @contextmanager
    def context_video(self):
        for self.video in self.video_list:
            yield

    @contextmanager
    def context_name(self):
        for self.name in self.name_list:
            yield

    @contextmanager
    def context_proj(self):
        for self.proj in self.proj_list:
            yield

    @contextmanager
    def context_tiling(self):
        for self.tiling in self.tiling_list:
            yield

    @contextmanager
    def context_quality(self):
        for self.quality in self.quality_list:
            yield

    @contextmanager
    def context_tile(self):
        for self.tile in self.tile_list:
            yield

    @contextmanager
    def context_chunk(self):
        for self.chunk in self.chunk_list:
            yield


class GlobalPaths(Factors):
    overwrite = False
    dectime_folder = Path('dectime')
    segment_folder = Path('segment')
    graphs_folder = Path('graphs')
    original_folder = Path('original')
    lossless_folder = Path('lossless')
    compressed_folder = Path('compressed')
    viewport_folder = Path('viewport')
    quality_folder = Path('quality')
    siti_folder = Path('siti')
    check_folder = Path('check')
    operation_folder = Path('')

    @property
    def project_path(self) -> Path:
        return Path('../results') / self.config['project']

    @property
    def dectime_result_json(self) -> Path:
        """
        By Video
        :return:
        """
        folder = self.project_path / self.dectime_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'time_{self.video}.json'

    @property
    def bitrate_result_json(self) -> Path:
        folder = self.project_path / self.segment_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'rate_{self.video}.json'

    @property
    def quality_result_json(self) -> Path:
        folder = self.project_path / self.quality_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'quality_{self.video}.json'


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


class Utils(Log, ContextObj):
    command_pool: list

    def __init__(self, config: str):
        self.config = Config(config)
        self.print_resume()
        start = time()
        with self.logger():
            self.main()
        print(f"\n\tTotal time={time() - start}.")

    def main(self):
        ...

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

    def clear_state(self):
        self.metric = None
        self._video = None
        self._proj = None
        self._name = None
        self._tiling = None
        self._quality = None
        self._tile = None
        self._chunk = None

    @property
    def state(self):
        s = []
        if self.proj is not None:
            s.append(self.proj)
        if self.name is not None:
            s.append(self.name)
        if self.tiling is not None:
            s.append(self.tiling)
        if self.quality is not None:
            s.append(self.quality)
        if self.tile is not None:
            s.append(self.tile)
        if self.chunk is not None:
            s.append(self.chunk)
        return s

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.video_list)} videos:\n'
              f'  operation: {self.__class__.__name__}\n'
              f'  project: {self.project}\n'
              f'  codec: {self.codec}\n'
              f'  fps: {self.fps}\n'
              f'  gop: {self.gop}\n'
              f'  qualities: {self.quality_list}\n'
              f'  patterns: {self.tiling_list}')
        print('=' * 70)

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

    @contextmanager
    def multi(self):
        self.command_pool = []
        try:
            yield
            with Pool(4) as p:
                p.map(run_command, self.command_pool)  # for command in self.command_pool:  #     run_command(command)
        finally:
            pass
