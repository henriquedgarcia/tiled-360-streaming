from contextlib import contextmanager
from multiprocessing import Pool
from time import time
from pathlib import Path

from lib.assets.logger import Logger
from lib.assets.config import config
from lib.util import splitx, run_command


class Worker(Logger):
    command_pool: list

    def __init__(self, config_file: Path, videos_file: Path):
        self.config = config.set_config(config_file, videos_file)
        self.log = Logger()
        self.print_resume()
        start = time()

        with self.log.logger():
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
                p.map(run_command,
                      self.command_pool)  # for command in self.command_pool:  #     run_command(command)
        finally:
            pass
