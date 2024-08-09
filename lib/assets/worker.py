from contextlib import contextmanager
from multiprocessing import Pool
from time import time

from config.config import config
from lib.assets.logger import logger
from lib.utils.util import run_command
from .context import ctx


class Worker:
    command_pool: list

    def __init__(self):
        self.print_resume()
        start = time()

        with logger.logger_context():
            self.main()
        print(f"\n\tTotal time={time() - start}.")

    def main(self):
        ...

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(ctx.name_list)} videos:\n'
              f'  operation: {self.__class__.__name__}\n'
              f'  project_folder: {config.project_folder}\n'
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
