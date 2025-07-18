import os
from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.progressbar import ProgressBar
from lib.make_tiles_seen import MakeTilesSeen
from lib.utils.context_utils import task
from lib.utils.util import load_json, print_error


class GetTilesSeen(MakeTilesSeen):
    tiles_seen_result: list
    progress_bar: ProgressBar
    cools_names: list

    def init(self):
        self.cools_names = ['name', 'projection', 'tiling', 'user', 'chunk', 'frame', 'tiles_seen']

    def main(self):
        for _ in self.iterate_name_projection:
            with task(self):
                self.check_seen_tiles_result_by_name()
                self.make_tiles_seen_result()
        self.merge()

    def check_seen_tiles_result_by_name(self):
        try:
            df = pd.read_pickle(self.seen_tiles_result_by_name)
            if df.size == 270000:
                raise AbortError('Seen tiles JSON is OK.')
        except FileNotFoundError:
            pass

    def make_tiles_seen_result(self):
        total = len(self.tiling_list) * len(self.users_list_by_name)

        tiles_seen_result = []
        for n in self.iterate_tiling_user():
            print(f'\rProcessing {n}/{total} - {self.ctx}', end='')
            try:
                user_seen_tiles = load_json(self.user_seen_tiles_json)['frames']
            except FileNotFoundError:
                self.logger.register_log('User seen tiles JSON is MISSING', self.user_seen_tiles_json)
                raise AbortError('\nUser seen tiles JSON is MISSING.')

            for frame in range(1800):
                tiles_seen = user_seen_tiles[frame]
                tiles_seen = list(map(int, tiles_seen))
                key = (self.name, self.projection, self.tiling, int(self.user), frame // 30, frame % 30, tiles_seen)
                tiles_seen_result.append(key)

        print('\nSaving')
        df = pd.DataFrame(tiles_seen_result, columns=self.cools_names)
        df.set_index(self.cools_names[:-1], inplace=True)
        df.to_pickle(self.seen_tiles_result_by_name)

    def merge(self):
        merged = None
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                df = pd.read_pickle(self.seen_tiles_result_by_name)
                merged = (df if merged is None
                          else pd.concat([merged, df], axis=0))
        if merged.size != 2160000:
            print_error('Dataframe size mismatch.')
            raise AbortError

        merged.to_pickle(self.seen_tiles_result)


if __name__ == '__main__':
    os.chdir('../')

    files = ['config_cmp_qp.json', 'config_erp_qp.json', 'config_erp_crf.json', 'config_erp_crf.json']
    videos_files = ['videos_reduced.json', 'videos_reversed.json', 'videos_lumine.json', 'videos_container0.json', 'videos_container1.json',
                    'videos_fortrek.json', 'videos_hp_elite.json', 'videos_alambique.json', 'videos_test.json',
                    'videos_full.json']

    config_file = Path('config') / files[0]
    videos_file = Path('config') / videos_files[0]

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    GetTilesSeen(ctx)
