from contextlib import contextmanager
from typing import Any

from pywin.mfc.object import Object

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.worker import Worker, ProgressBar
from lib.get_tiles import GetTilesPaths
from lib.utils.worker_utils import get_nested_value
from lib.utils.worker_utils import save_json, load_json, print_error


class GetGetTiles(Worker, CtxInterface):
    get_tiles_result: AutoDict
    get_tiles_paths: GetTilesPaths

    def iter_proj_tiling_tile_qlt_chunk(self):
        for self.projection in self.projection_list:
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    self.ctx.iterations += 1
                    yield

    total: int

    def init(self):
        self.get_tiles_paths = GetTilesPaths(self.ctx)
        self.total = (181 * len(self.projection_list)
                      * len(self.quality_list) * len(self.chunk_list))

    @contextmanager
    def task(self):
        class_name = self.__class__.__name__
        print(f'==== {class_name} {self.ctx} ====')
        self.get_tiles_result = AutoDict()
        t = ProgressBar(total=self.total, desc=class_name)

        try:
            for _ in self.iter_proj_tiling_tile_qlt_chunk():
                t.update(f'{self.ctx}')
                yield

        except FileNotFoundError as e:
            print_error('Chunk not Found.')
        except AbortError as e:
            print_error(f'\t{e.args[0]}')

        save_json(self.get_tiles_result, self.get_tiles_paths.get_tiles_result_json)
        del t

    def main(self):
        for self.name in self.name_list:
            if self.get_tiles_paths.get_tiles_result_json.exists():
                print_error(f'\tThe get_tiles_result_json exist.')
                continue

            with self.task():
                user_tiles_seen = load_json(self.get_tiles_paths.user_tiles_seen_json)
                self.set_get_tiles(user_tiles_seen)

            save_json(self.get_tiles_result, self.get_tiles_paths.get_tiles_result_json)

    def set_get_tiles(self, value: Any):
        if isinstance(value, dict):
            keys = [self.name, self.projection, self.tiling, self.user]
            result = get_nested_value(self.get_tiles_result, keys)
            result.update(value)
