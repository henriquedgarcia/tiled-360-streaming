import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from multiprocessing import Pool

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.logger import Logger
from lib.assets.status_ctx import StatusCtx
from lib.utils.util import run_command


class Multi(ABC):
    command_pool: list

    @contextmanager
    def pool(self):
        self.command_pool = []
        try:
            yield
            with Pool(4) as p:
                p.map(subprocess.run, self.command_pool)
        finally:
            pass


class Worker(ABC, CtxInterface):
    def __init__(self, ctx: Context):
        self.ctx = ctx

    logger: Logger
    status: StatusCtx

    def run(self):
        self.logger = Logger(self.ctx)
        self.status = StatusCtx(self.ctx)
        self.print_resume()

        with self.logger.logger_context(self.__class__.__name__):
            with self.status.status_context(self.__class__.__name__):
                self.init()
                self.main()

    def __str__(self):
        return str(self.ctx)

    @abstractmethod
    def main(self):
        ...

    def init(self):
        pass

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

    @property
    def iterate_name_projection(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                yield

    @property
    def iterate_name_projection_metric(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.metric in self.metric_list:
                    self.ctx.iterations += 1
                    yield

    def iterate_name_projection_user(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.user in self.users_list_by_name:
                    self.ctx.iterations += 1
                    yield

    @property
    def iterate_name_projection_tiling(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    self.ctx.iterations += 1
                    yield

    def iterate_name_projection_tiling_user(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.user in self.users_list_by_name:
                        self.ctx.iterations += 1
                        yield

    def iterate_name_projection_tiling_tile(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        self.ctx.iterations += 1
                        yield

    @property
    def iterate_name_projection_tiling_tile_quality(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        for self.quality in self.quality_list:
                            self.ctx.iterations += 1
                            yield

    @property
    def iterate_name_projection_tiling_quality_user_chunk(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        for self.user in self.users_list_by_name:
                            for self.chunk in self.chunk_list:
                                self.ctx.iterations += 1
                                yield

    def iterate_name_projection_tiling_tile_quality_chunk(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        for self.quality in self.quality_list:
                            for self.chunk in self.chunk_list:
                                self.ctx.iterations += 1
                                yield

    def iterate_tiling_tile(self):
        for self.tiling in self.tiling_list:
            for self.tile in self.tile_list:
                self.ctx.iterations += 1
                yield

    @property
    def iterate_tiling_tile_quality_chunk(self):
        n = 0
        for self.tiling in self.tiling_list:
            for self.tile in self.tile_list:
                for self.quality in self.quality_list:
                    for self.chunk in self.chunk_list:
                        n += 1
                        yield n

    @property
    def iterate_tiling_quality_user(self):
        n = 0
        for self.tiling in self.tiling_list:
            for self.quality in self.quality_list:
                for self.user in self.users_list_by_name:
                    for self.chunk in self.chunk_list:
                        n += 1
                        yield n

    def iterate_tiling_user(self):
        n = 0
        for self.tiling in self.tiling_list:
            for self.user in self.users_list_by_name:
                self.ctx.iterations += 1
                n += 1
                yield n

    @property
    def iterate_user_chunks(self):
        for self.user in self.users_list_by_name:
            for self.chunk in self.chunk_list:
                self.ctx.iterations += 1
                yield
