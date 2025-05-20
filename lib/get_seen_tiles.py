import os
from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.progressbar import ProgressBar
from lib.make_tiles_seen import MakeTilesSeen
from lib.utils.context_utils import task
from lib.utils.util import load_json, save_pickle, print_error


class GetTilesSeen(MakeTilesSeen):
    tiles_seen_result: list
    progress_bar: ProgressBar

    def init(self):
        pass

    def main(self):
        for _ in self.iterate_name_projection:
            with task(self):
                self.check_seen_tiles_result_by_name()
                self.make_tiles_seen_result()
                self.save_tiles_seen()
        self.merge()

    def merge(self):
        if self.seen_tiles_result.exists():
            print('seen_tiles_result is OK.')
            return

        merged = None
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                df = pd.read_pickle(self.seen_tiles_result_by_name)
                merged = (df if merged is None
                          else pd.concat([merged, df], axis=0))
            if merged.size != 72000:
                print_error('Dataframe size mismatch.')
                raise AbortError

        merged.to_pickle(self.seen_tiles_result)

    def check_seen_tiles_result_by_name(self):
        try:
            size = self.seen_tiles_result_by_name.stat().st_size
        except FileNotFoundError:
            return

        if size < 10:
            self.seen_tiles_result_by_name.unlink(missing_ok=True)
            return

        raise AbortError('Seen tiles JSON is OK.')

    def make_tiles_seen_result(self):
        self.tiles_seen_result = []
        self.progress_bar = ProgressBar(len(self.projection_list) * len(self.tiling_list) * 30, desc=self.name)
        for _ in self.iterate_tiling_user():
            self.progress_bar.update(f'{self.ctx}')
            self.process_user_seen_tiles_json()

    def process_user_seen_tiles_json(self):
        user_seen_tiles = load_json(self.user_seen_tiles_json)['chunks']
        for self.chunk in self.chunk_list:
            seen_tiles = list(map(int, user_seen_tiles[self.chunk]))
            key = (self.name, self.projection, self.tiling,
                   int(self.user), int(self.chunk) - 1, seen_tiles)
            self.tiles_seen_result.append(key)

    def save_tiles_seen(self):
        columns = ['name', 'projection', 'tiling', 'user', 'chunk', 'seen_tiles']
        result = pd.DataFrame(self.tiles_seen_result, columns=columns)
        result.set_index(columns[:-1], inplace=True)
        save_pickle(result, self.seen_tiles_result_by_name)
        print('finished')


if __name__ == '__main__':
    os.chdir('../')

    # config_file = 'config_erp_qp.json'
    # config_file = 'config_cmp_crf.json'
    # config_file = 'config_erp_crf.json'
    # videos_file = 'videos_reversed.json'
    # videos_file = 'videos_lumine.json'
    # videos_file = 'videos_container0.json'
    # videos_file = 'videos_container1.json'
    # videos_file = 'videos_fortrek.json'
    # videos_file = 'videos_hp_elite.json'
    # videos_file = 'videos_alambique.json'
    # videos_file = 'videos_test.json'
    # videos_file = 'videos_full.json'

    config_file = Path('config/config_erp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    GetTilesSeen(ctx)
