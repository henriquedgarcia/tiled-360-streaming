from abc import ABC, abstractmethod
from contextlib import contextmanager
from multiprocessing import Pool

from config.config import Config
from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.logger import Logger
from lib.assets.status_ctx import StatusCtx
from lib.utils.worker_utils import run_command


class Multi(ABC):
    command_pool: list

    @contextmanager
    def pool(self):
        self.command_pool = []
        try:
            yield
            with Pool(4) as p:
                p.map(run_command, self.command_pool)
                # for command in self.command_pool:
                #     run_command(command)
        finally:
            pass


class Worker(ABC, CtxInterface):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.logger = Logger(ctx)
        self.status = StatusCtx(ctx)
        self.print_resume()

        with self.logger.logger_context(self.__class__.__name__):
            with self.status.status_context(self.__class__.__name__):
                self.main()

    @abstractmethod
    def main(self):
        ...

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.ctx.name_list)} videos:\n'
              f'  operation: {self.__class__.__name__}\n'
              f'  project_folder: {self.config.project_folder}\n'
              f'  fps: {self.config.fps}\n'
              f'  gop: {self.config.gop}\n'
              f'  qualities: {self.ctx.quality_list}\n'
              f'  patterns: {self.ctx.tiling_list}')
        print('=' * 70)
