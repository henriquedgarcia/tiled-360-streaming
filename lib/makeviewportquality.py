import json
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from py360tools import CMP
from py360tools.utils import LazyProperty
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from lib.assets.autodict import AutoDict
from lib.assets.chunkprojectionreader import ChunkProjectionReader
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.paths.make_tiles_seen_paths import TilesSeenPaths
from lib.assets.paths.make_chunk_quality_paths import MakeChunkQualityPaths
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import save_json, load_json, get_nested_value, print_error, save_pickle

Tiling = str
Tile = str
NumpyArray = np.ndarray


class Props(Worker, ViewportQualityPaths,
            TilesSeenPaths, MakeChunkQualityPaths, ABC):
    seen_tiles_deg_path: dict[Tile, Path]
    seen_tiles_ref_path: dict[Tile, Path]
    seen_tiles_result: Optional[dict]
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

    @property
    def seen_tiles(self):
        """
        depends [self.name, self.projection, self.tiling, self.user, self.chunk]
        need make: self.seen_tiles_result = load_json(self.seen_tiles_result_json)
            self.seen_tiles_result_json depends [self.name, self.fov]
        :return:
        """
        keys = [self.name, self.projection, self.tiling, self.user]
        return get_nested_value(self.seen_tiles_result, keys)['chunks'][self.chunk]

    @LazyProperty
    def hmd_dataset(self) -> dict[str, dict[str, list]]:
        return load_json(self.config.dataset_file)

    @property
    def user_hmd_data(self):
        return self.hmd_dataset[self.name + '_nas'][self.user]


class ViewportQuality(Props):
    def init(self):
        self.projection = 'cmp'

    viewport_frame_ref_3dArray: Optional[np.ndarray] = None

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

        for self.tiling in self.tiling_list:
            self.proj_obj = CMP(tiling=self.tiling, proj_res='3240x2160',
                                vp_res='1320x1080', fov_res='110x90')
            for self.name in self.name_list:
                self.seen_tiles_result = load_json(self.seen_tiles_result_json)
                for self.user in self.users_list_by_name:
                    for self.chunk in self.chunk_list:
                        self.viewport_frame_ref_3dArray = None

                        for self.quality in self.quality_list:
                            if self.check_json(): continue

                            self.make_viewport_frame_ref_3dArray()
                            results = self.calc_chunk_error_per_frame()
                            save_json(results, self.user_viewport_quality_json)

    def check_json(self):
        def check():
            size = self.user_viewport_quality_json.stat().st_size
            if size < 10:
                raise FileNotFoundError
            msg = f'{self.ctx}. File exists. skipping.'
            print_error(f'{msg:<90}')

        try:
            check()
        except FileNotFoundError:
            return False
        return True

    def calc_chunk_error_per_frame(self, ):
        deg_tiles_path = {self.tile: self.decodable_chunk for self.tile in self.seen_tiles}
        deg_proj_frame_vreader = ChunkProjectionReader(deg_tiles_path, proj=self.proj_obj)
        results = defaultdict(list)
        for self.frame, yaw_pitch_roll in enumerate(self.chunk_yaw_pitch_roll_per_frame):
            msg = f'{self.ctx}.'
            print(f'\r{msg:<90}', end='')
            viewport_frame_deg = deg_proj_frame_vreader.extract_viewport(yaw_pitch_roll)

            _mse = mse(self.viewport_frame_ref_3dArray[self.frame], viewport_frame_deg)
            _ssim = ssim(self.viewport_frame_ref_3dArray[self.frame], viewport_frame_deg, data_range=255.0,
                         gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            results['mse'].append(_mse)
            results['ssim'].append(_ssim)
        self.frame = None
        print('')
        return results

    def make_viewport_frame_ref_3dArray(self):
        if self.viewport_frame_ref_3dArray is not None:
            return
        ref_tiles_path = {self.tile: self.reference_chunk for self.tile in self.seen_tiles}
        ref_proj_frame_vreader = ChunkProjectionReader(ref_tiles_path, proj=self.proj_obj)
        self.viewport_frame_ref_3dArray = np.zeros((len(self.chunk_yaw_pitch_roll_per_frame), 1080, 1320))
        for self.frame, yaw_pitch_roll in enumerate(self.chunk_yaw_pitch_roll_per_frame):
            self.viewport_frame_ref_3dArray[self.frame] = ref_proj_frame_vreader.extract_viewport(yaw_pitch_roll)
        self.frame = None


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
                            if self.check_json():
                                ok += 1
                                continue
                            miss += 1

                    if miss == 0:
                        print(' OK!')
                        continue
                    print(f' Chunks que faltam: {miss}, Chunks ok: {ok}. Total: {miss + ok}')
                    miss_total += miss
                    ok_total += ok

                    result[self.name][self.tiling][self.quality]['ok'] = ok
                    result[self.name][self.tiling][self.quality]['miss'] = miss
                    # print('')
        print(json.dumps(result, indent=2))
        print(f'\nChunks que faltam: {miss_total}, Chunks ok: {ok_total}. Total: {miss_total + ok_total}')
        Path(f'CheckViewportQuality.json').write_text(json.dumps(result, indent=2))


class GetViewportQuality(ViewportQuality):
    def main(self):

        typos = {'name': str, 'projection': str, 'tiling': str,
                 'quality': int, 'user': int, 'chunk': int, 'frame': int,
                 'mse': float, 'ssim': float}
        for self.name in self.name_list:
            new_data = []
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        for self.user in self.users_list_by_name:
                            frame = 0
                            for self.chunk in self.chunk_list:
                                print(f'\r{self.name}_{self.tiling}_qp{self.quality}_user{self.user}_chunk{self.chunk}', end='')
                                # user_viewport_quality é um dicionário de listas. As chaves são 'ssim' e 'mse'.
                                # As listas contem as métricas usando float64 para cada frame do chunk (30 frames)
                                user_viewport_quality = load_json(self.user_viewport_quality_json)
                                mse_ = user_viewport_quality['mse']
                                ssim_ = user_viewport_quality['ssim']
                                for (m, s) in zip(mse_, ssim_):
                                    data = (self.name, self.projection, self.tiling, int(self.quality), int(self.user), int(self.chunk) - 1, frame, m, s)
                                    new_data.append(data)
                                    frame += 1
                            # break
            keys = list(typos.keys())

            df = pd.DataFrame(new_data, columns=keys)
            df.set_index(keys[:-2], inplace=True)
            save_pickle(df, self.user_viewport_quality_result_pickle)
