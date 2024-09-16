from collections import Counter
from time import time

import numpy as np
from matplotlib import pyplot as plt
from py360tools import ERP, CMP

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.assets.context import Context
from lib.assets.errors import GetTilesOkError, HMDDatasetError
from lib.assets.logger import Logger
from lib.assets.paths.gettilespaths import get_tiles_paths
from lib.assets.paths.segmenterpaths import segmenter_paths
from lib.utils.util import load_json, save_json, splitx, print_error


class GetTiles:
    projection_dict: AutoDict

    def __init__(self, ctx: Context, config: Config, logger: Logger):
        self.ctx = ctx
        self.config = config
        self.logger = logger
        self.hmd_dataset = load_json(get_tiles_paths.dataset_json)

        self.create_projections_obj()
        self.process()

    def create_projections_obj(self):
        self.projection_dict = AutoDict()
        for tiling in self.ctx.tiling_list:
            erp = ERP(tiling=tiling, proj_res='1080x540', vp_res='660x540', fov_res=self.config.fov)
            cmp = CMP(tiling=tiling, proj_res='810x540', vp_res='660x540', fov_res=self.config.fov)
            self.projection_dict['erp'][tiling] = erp
            self.projection_dict['cmp'][tiling] = cmp

    def assert_getTiles(self):
        if self.logger.get_status('get_tiles_ok'):
            print(f'\t{self.ctx.name} is OK. Skipping')
            raise GetTilesOkError('Get tiles is OK.')

    def process(self):
        for self.ctx.name in self.ctx.name_list:
            for self.ctx.projection in self.ctx.projection_list:
                for self.ctx.tiling in self.ctx.tiling_list:
                    for self.ctx.user in self.ctx.user_list:
                        self.for_each_user()

    def for_each_user(self):
        print(f'==== GetTiles {self.ctx} ====')
        try:
            self.get_tiles_by_video()
            # self.count_tiles()
            # self.heatmap()
            # self.plot_test()
        except (HMDDatasetError,) as e:
            print_error(f'\t{e.args[0]}')

    def get_tiles_by_video(self):
        start = time()

        user_hmd_data = self.get_user_hmd_data()
        if user_hmd_data == {}:
            self.logger.register_log(f'HMD samples is missing, user{self.ctx.user}')
            raise HMDDatasetError(f'HMD samples is missing, user{self.ctx.user}')

        tiles_seen_by_frame = self.get_tiles_seen_by_frame(user_hmd_data)
        tiles_seen_by_chunk = self.get_tiles_seen_by_chunk(tiles_seen_by_frame)

        tiles_seen = {'frames': tiles_seen_by_frame,
                      'chunks': tiles_seen_by_chunk}

        save_json(tiles_seen, get_tiles_paths.user_tiles_seen_json)
        print(f'\ttime =  {time() - start}')

    def get_tiles_seen_by_frame(self, user_hmd_data):
        if self.ctx.tiling == '1x1':
            return [["0"]] * self.config.n_frames

        tiles_seen_by_frame = []
        projection_obj = self.projection_dict[self.ctx.projection][self.ctx.tiling]

        for frame, yaw_pitch_roll in enumerate(user_hmd_data, 1):
            print(f'\r\tframe {frame:04d}/{self.config.n_frames}', end='')
            vptiles: list[str] = projection_obj.get_vptiles(yaw_pitch_roll)
            vptiles = list(map(str, map(int, vptiles)))
            tiles_seen_by_frame.append(vptiles)
        return tiles_seen_by_frame

    def get_tiles_seen_by_chunk(self, tiles_seen_by_frame):
        if self.ctx.tiling == '1x1':
            return {str(i): ["0"] for i in range(1, int(self.config.duration) + 1)}

        tiles_seen_by_chunk = {}
        tiles_in_chunk = set()
        for frame, vptiles in enumerate(tiles_seen_by_frame):
            tiles_in_chunk.update(vptiles)

            if (frame + 1) % 30 == 0:
                chunk_id = frame // 30 + 1  # chunk start from 1
                tiles_seen_by_chunk[f'{chunk_id}'] = list(tiles_in_chunk)
                tiles_in_chunk.clear()
        return tiles_seen_by_chunk

    def get_user_hmd_data(self):
        user_hmd_data = self.hmd_dataset[self.ctx.name + '_nas'][self.ctx.user]
        if user_hmd_data == {}:
            print_error(f'\tHead movement user samples are missing.')
            self.logger.register_log(f'HMD samples is missing, user{self.ctx.user}', self.ctx.name)
            self.ctx.error_flag = True
            raise
        return self.hmd_dataset[self.ctx.name + '_nas'][self.ctx.user]

    def load_results(self):
        try:
            self.ctx.results = load_json(segmenter_paths.get_tiles_json,
                                         object_hook=AutoDict)
        except FileNotFoundError:
            self.ctx.results = AutoDict()

    def save_results(self):
        segmenter_paths.get_tiles_json.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.ctx.results, segmenter_paths.get_tiles_json)

    def heatmap(self):
        results = load_json(segmenter_paths.counter_tiles_json)

        for self.ctx.tiling in self.ctx.tiling_list:
            if self.ctx.tiling == '1x1': continue

            filename = (f'heatmap_tiling_{self.config.hmd_dataset_name}_{self.ctx.projection}_{self.ctx.name}_'
                        f'{self.ctx.tiling}_fov{self.config.fov}.png')
            heatmap_tiling = (segmenter_paths.get_tiles_folder / filename)
            if heatmap_tiling.exists(): continue

            tiling_result = results[self.ctx.tiling]

            h, w = splitx(self.ctx.tiling)[::-1]
            grade = np.zeros((h * w,))

            for item in tiling_result: grade[int(item)] = tiling_result[item]
            grade = grade.reshape((h, w))

            fig, ax = plt.subplots()
            im = ax.imshow(grade, cmap='jet', )
            ax.set_title(f'Tiling {self.ctx.tiling}')
            fig.colorbar(im, ax=ax, label='chunk frequency')

            # fig.show()
            fig.savefig(f'{heatmap_tiling}')

    def count_tiles(self):
        if segmenter_paths.counter_tiles_json.exists(): return

        self.ctx.results = load_json(segmenter_paths.get_tiles_json)
        result = {}

        for self.ctx.tiling in self.ctx.tiling_list:
            if self.ctx.tiling == '1x1': continue

            # <editor-fold desc="Count tiles">
            tiles_counter_chunks = Counter()  # Collect tiling count

            for self.ctx.user in self.ctx.user_list:
                result_chunks = self.ctx.results[self.ctx.projection][self.ctx.name]
                result_chunks = result_chunks[self.ctx.tiling][self.ctx.user]['chunks']

                for chunk in result_chunks:
                    tiles_counter_chunks = tiles_counter_chunks + Counter(result_chunks[chunk])
            # </editor-fold>

            print(tiles_counter_chunks)
            dict_tiles_counter_chunks = dict(tiles_counter_chunks)

            # <editor-fold desc="Normalize Counter">
            nb_chunks = sum(dict_tiles_counter_chunks.values())
            for self.ctx.tile in self.ctx.tile_list:
                try:
                    dict_tiles_counter_chunks[self.ctx.tile] /= nb_chunks
                except KeyError:
                    dict_tiles_counter_chunks[self.ctx.tile] = 0
            # </editor-fold>

            result[self.ctx.tiling] = dict_tiles_counter_chunks

        save_json(result, segmenter_paths.counter_tiles_json)
