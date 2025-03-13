from contextlib import contextmanager
from typing import Any

import pandas as pd

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.get_tiles import GetTilesPaths
from lib.utils.util import print_error, save_json, load_json, get_nested_value, save_pickle


class GetGetTiles(Worker, CtxInterface):
    get_tiles_paths: GetTilesPaths
    progress_bar: ProgressBar

    def iter_name_tiling_user(self):
        self.projection = 'cmp'
        self.progress_bar = ProgressBar(total=(len(self.name_list)
                                               * len(self.tiling_list)
                                               * 30),
                                        desc=self.__class__.__name__)
        for self.name in self.name_list:
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    self.progress_bar.update(f'{self.ctx}')
                    self.ctx.iterations += 1
                    yield

    def init(self):
        self.get_tiles_paths = GetTilesPaths(self.ctx)
        if self.get_tiles_paths.get_tiles_result_pickle.exists():
            print('file exists')
            exit(0)

    def main(self):
        get_tiles_result = []
        for _ in self.iter_name_tiling_user():
            user_tiles_seen = load_json(self.get_tiles_paths.user_tiles_seen_json)
            for self.chunk in self.chunk_list:
                get_tiles = user_tiles_seen['chunks'][self.chunk]
                get_tiles = list(map(int, get_tiles))
                get_tiles_result.append(
                    (self.name, self.projection, self.tiling, int(self.user),
                     int(self.chunk), get_tiles))

        result = pd.DataFrame(get_tiles_result, columns=['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'get_tiles'])
        result.set_index(['name', 'projection', 'tiling', 'tile', 'quality', 'chunk'], inplace=True)
        save_pickle(result['get_tiles'], self.get_tiles_paths.get_tiles_result_pickle)
        print('finished')
