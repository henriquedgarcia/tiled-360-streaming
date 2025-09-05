import json
import os
from abc import ABC
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from py360tools import CMP, ERP, Viewport
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.assets.chunkprojectionreader import ChunkProjectionReader
from lib.assets.context import Context
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import save_json, load_json, print_error

Tiling = str
NumpyArray = np.ndarray


class Props(Worker, ViewportQualityPaths, ABC):
    seen_tiles_deg_path: dict[str, Path]
    seen_tiles_ref_path: dict[str, Path]
    canvas: NumpyArray
    ui: ProgressBar
    get_tiles: dict
    proj_obj: CMP
    results: defaultdict

    @property
    def chunk_yaw_pitch_roll_per_frame(self) -> list[tuple[float, float, float]]:
        chunk = int(self.chunk) - 1
        start = chunk * 30
        return self.user_hmd_data[slice(start, start + 30)]

    # @property
    # def seen_tiles(self):
    #     """
    #     depends [self.name, self.projection, self.tiling, self.user, self.chunk]
    #     need make: self.tiles_seen_result = load_json(self.seen_tiles_result_json)
    #         self.seen_tiles_result_json depends [self.name, self.fov]
    #     :return:
    #     """
    #     keys = [self.name, self.projection, self.tiling, self.user]
    #     return get_nested_value(self.video_seen_tiles, keys)['chunks'][self.chunk]

    @cached_property
    def hmd_dataset(self) -> dict[str, dict[str, list]]:
        return load_json(self.config.dataset_file)

    @property
    def user_hmd_data(self):
        return self.hmd_dataset[self.name + '_nas'][self.user]


class ViewportQuality(Props):
    seen_tiles_db: DataFrame
    seen_tiles_level: list
    viewport_frame_ref_3dArray: Optional[np.ndarray]
    results: defaultdict
    video_seen_tiles: pd.DataFrame

    def init(self):
        pass

    def main(self):
        """
        User Viewport Frame Quality
        FrameResult = (frame_id, (mse, ssim))
        self.results = list of Result [FrameResult, ...]

        user_hmd_data: list[list[float, float, float]]
        user_hmd_data[frame] = [yaw, pitch, roll]
        yaw, pitch, roll in radians
        :return:
        """
        for self.projection in self.projection_list:
            self.seen_tiles_db = pd.read_pickle(self.seen_tiles_result)

            for self.name in self.name_list:
                for self.projection in self.projection_list:
                    for self.tiling in self.tiling_list:
                        self.proj_obj = (CMP if self.projection == 'cmp' else ERP)(tiling=self.tiling, proj_res=self.scale)
                        self.vp = Viewport('1080x1080', '90x90', projection=self.proj_obj)

                        for self.user in self.users_list_by_name:
                            for self.quality in self.quality_list:
                                for self.chunk in self.chunk_list:
                                    print(f'{self}. Processing... ', end='')
                                    try:
                                        self.calc_chunk_error_per_frame()
                                    except StopIteration:
                                        print_error(f'Decode error. Frame {self.frame}')
                                        self.logger.register_log(f'Decode error. Frame {self.frame}', self.user_viewport_quality_json)
                                        continue
                                    print(f'Saving...')
                                    save_json(self.results, self.user_viewport_quality_json)

    def get_seen_tiles(self) -> list[int]:
        seen_tiles_level = ('name', 'projection', 'tiling', 'user', 'chunk')
        seen_tiles = self.seen_tiles_db.xs((self.name, self.projection, self.tiling, int(self.user), int(self.chunk) - 1),
                                           level=seen_tiles_level)
        a = set()
        for item in list(seen_tiles['tiles_seen']):
            a.update(item)
        return list(a)

    vp: Viewport

    def make_proj_and_vp_obj(self):
        self.proj_obj = (CMP if self.projection == 'cmp' else ERP)(tiling=self.tiling, proj_res=self.scale)
        self.vp = Viewport('1080x1080', '90x90', projection=self.proj_obj)

    def calc_chunk_error_per_frame(self, ):
        ref_tiles_path = {self.tile: self.reference_chunk for self.tile in self.tile_list}
        ref_proj_frame_vreader = ChunkProjectionReader(ref_tiles_path, viewport=self.vp)

        deg_tiles_path = {self.tile: self.decodable_chunk for self.tile in self.get_seen_tiles()}
        deg_proj_frame_vreader = ChunkProjectionReader(deg_tiles_path, viewport=self.vp)

        self.results = defaultdict(list)
        for self.frame, yaw_pitch_roll in enumerate(self.chunk_yaw_pitch_roll_per_frame):
            viewport_frame_ref = ref_proj_frame_vreader.extract_viewport(yaw_pitch_roll)
            viewport_frame_deg = deg_proj_frame_vreader.extract_viewport(yaw_pitch_roll)
            _mse = mse(viewport_frame_ref, viewport_frame_deg)
            _ssim = ssim(viewport_frame_ref, viewport_frame_deg, data_range=255.0,
                         gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            self.results['mse'].append(_mse)
            self.results['ssim'].append(_ssim)
        self.frame = None


# from PILL import Image
# Image.fromarray(viewport_frame_ref).show()
# Image.fromarray(np.abs(viewport_frame_ref - viewport_frame_deg)).show()


class CheckViewportQuality(ViewportQuality):
    def main(self):
        result = AutoDict()
        miss_total = 0
        ok_total = 0

        for self.name in self.name_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:

                    print(f'\r{self.name}_{self.tiling}_{self.quality}  ', end='')
                    miss = 0
                    ok = 0
                    for self.user in self.users_list_by_name:
                        for self.chunk in self.chunk_list:
                            if self.user_viewport_quality_json.exists() and self.user_viewport_quality_json.stat().st_size > 10:
                                ok += 1
                                continue
                            miss += 1

                    if miss == 0:
                        print('OK!')
                        continue
                    print(f'\n\tChunks que faltam: {miss}, Chunks ok: {ok}. Total: {miss + ok}')
                    miss_total += miss
                    ok_total += ok

                    result[self.name][self.tiling][self.quality]['ok'] = ok
                    result[self.name][self.tiling][self.quality]['miss'] = miss
                    # print('')
        print(json.dumps(result, indent=2))
        print(f'\nChunks que faltam: {miss_total}, Chunks ok: {ok_total}. Total: {miss_total + ok_total}')
        Path(f'CheckViewportQuality.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    os.chdir('../')

    config_file = Path('config/config_pres_qp.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    ViewportQuality(ctx).run()
