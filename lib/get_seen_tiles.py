import pandas as pd

from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.assets.paths.make_tiles_seen_paths import TilesSeenPaths
from lib.utils.util import load_json, save_pickle, print_error


class GetSeenTiles(Worker, TilesSeenPaths):
    seen_tiles_result: list
    progress_bar: ProgressBar

    def init(self):
        if self.seen_tiles_result_pickle.exists():
            print_error(f'{self.seen_tiles_result_pickle} exists')
            return

        self.seen_tiles_result = []
        self.progress_bar = ProgressBar(total=(len(self.name_list) * len(self.projection_list)
                                               * len(self.tiling_list) * 30),
                                        desc=self.__class__.__name__)

    def main(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.user in self.users_list_by_name:
                        self.progress_bar.update(f'{self.ctx}')
                        self.process_user_seen_tiles_json()

        columns = ['name', 'projection', 'tiling', 'user', 'chunk', 'seen_tiles']
        result = pd.DataFrame(self.seen_tiles_result, columns=columns)
        result.set_index(columns[:-1], inplace=True)
        save_pickle(result, self.seen_tiles_result_pickle)
        print('finished')

    def process_user_seen_tiles_json(self):
        user_seen_tiles = load_json(self.user_seen_tiles_json)['chunks']
        for self.chunk in self.chunk_list:
            seen_tiles = list(map(int, user_seen_tiles[self.chunk]))
            key = (self.name, self.projection, self.tiling,
                   int(self.user), int(self.chunk) - 1, seen_tiles)
            self.seen_tiles_result.append(key)


