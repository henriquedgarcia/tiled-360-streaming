from contextlib import contextmanager
from multiprocessing import Pool
from time import time
from pathlib import Path

from .context import ctx
from lib.assets.logger import logger
from lib.assets.config import config
from lib.utils.util import splitx, run_command


def tile_position():
    """
    Need video, tiling and tile
    :return: x1, x2, y1, y2
    """
    proj_h, proj_w = config.video_shape
    tiling_w, tiling_h = splitx(ctx.tiling)
    tile_w, tile_h = int(proj_w / tiling_w), int(proj_h / tiling_h)
    tile_m, tile_n = int(ctx.tile) % tiling_w, int(ctx.tile) // tiling_w
    x1 = tile_m * tile_w
    y1 = tile_n * tile_h
    x2 = tile_m * tile_w + tile_w  # not inclusive [...)
    y2 = tile_n * tile_h + tile_h  # not inclusive [...)
    return x1, y1, x2, y2


class Worker:
    command_pool: list

    def __init__(self, config_file: Path, videos_file: Path):
        self.config = config.set_config(config_file, videos_file)
        self.print_resume()
        start = time()

        with logger.logger_context():
            self.main()
        print(f"\n\tTotal time={time() - start}.")

    def main(self):
        ...

    @property
    def state(self):
        s = []
        if ctx.proj is not None:
            s.append(ctx.proj)
        if ctx.name is not None:
            s.append(ctx.name)
        if ctx.tiling is not None:
            s.append(ctx.tiling)
        if ctx.quality is not None:
            s.append(ctx.quality)
        if ctx.tile is not None:
            s.append(ctx.tile)
        if ctx.chunk is not None:
            s.append(ctx.chunk)
        return s

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(ctx.video_list)} videos:\n'
              f'  operation: {self.__class__.__name__}\n'
              f'  project: {config.project}\n'
              f'  codec: {config.codec}\n'
              f'  fps: {config.fps}\n'
              f'  gop: {config.gop}\n'
              f'  qualities: {ctx.quality_list}\n'
              f'  patterns: {ctx.tiling_list}')
        print('=' * 70)

    @contextmanager
    def multi(self):
        self.command_pool = []
        try:
            yield
            with Pool(4) as p:
                p.map(run_command, self.command_pool)
                # for command in self.command_pool:
                #     run_command(command)
        finally:
            pass
