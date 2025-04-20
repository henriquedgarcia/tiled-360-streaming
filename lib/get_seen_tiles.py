import pandas as pd

from lib.assets.ctxinterface import CtxInterface
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.get_tiles import SeenTilesPaths
from lib.utils.util import load_json, save_pickle, print_error


class GetSeenTiles(Worker, CtxInterface):
    seen_tiles_paths: SeenTilesPaths
    progress_bar: ProgressBar

    def init(self):
        self.seen_tiles_paths = SeenTilesPaths(self.ctx)
        self.projection = 'cmp'

        self.progress_bar = ProgressBar(total=(len(self.name_list)
                                               * len(self.tiling_list)
                                               * 30),
                                        desc=self.__class__.__name__)

    def main(self):
        for self.name in self.name_list:
            if self.seen_tiles_paths.seen_tiles_result_pickle.exists():
                print_error(f'{self.seen_tiles_paths.seen_tiles_result_pickle} exists')
                continue

            self.seen_tiles_result = []

            for self.tiling in self.tiling_list:
                for self.user in self.users_list_by_name:
                    self.progress_bar.update(f'{self.ctx}')
                    self.process_user_seen_tiles()

            result = pd.DataFrame(self.seen_tiles_result, columns=['name', 'projection', 'tiling', 'user', 'chunk', 'seen_tiles'])
            result.set_index(['name', 'projection', 'tiling', 'user', 'chunk'], inplace=True)
            save_pickle(result['seen_tiles'], self.seen_tiles_paths.seen_tiles_result_pickle)
            print('finished')

    def process_user_seen_tiles(self):
        user_seen_tiles = load_json(self.seen_tiles_paths.user_seen_tiles_json)
        for self.chunk in self.chunk_list:
            seen_tiles = self.get_seen_tiles(user_seen_tiles)
            self.set_seen_tiles(self.seen_tiles_result, seen_tiles)

    def get_seen_tiles(self, user_seen_tiles):
        seen_tiles = user_seen_tiles['chunks'][self.chunk]
        return list(map(int, seen_tiles))

    def set_seen_tiles(self, seen_tiles_result, seen_tiles):
        key = [self.name, self.projection, self.tiling,
               int(self.user),
               int(self.chunk) - 1, seen_tiles]
        seen_tiles_result.append(key)
