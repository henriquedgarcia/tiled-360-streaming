from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Iterable

import numpy as np
from py360tools import ProjectionBase
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from lib.assets.autodict import AutoDict
from lib.assets.mountframe import MountFrame
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.paths.seen_tiles_paths import SeenTilesPaths
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import build_projection, save_json, load_json, get_nested_value, set_nested_value


class Props(Worker, ViewportQualityPaths, MakeDecodablePaths,
            SeenTilesPaths, ChunkQualityPaths, ABC):
    results: dict

    proj_obj: ProjectionBase
    yaw_pitch_roll_dict_iter: dict[str, Iterator]

    get_tiles: dict
    seen_tiles: list[str]
    yaw_pitch_roll_by_frame: list[int]
    seen_tiles_deg_path: dict[str, Path]
    seen_tiles_ref_path: dict[str, Path]
    canvas: np.ndarray
    ui: ProgressBar
    projection = 'cmp'
    get_tiles: dict
    proj_obj: dict[str, ProjectionBase]
    results: defaultdict
    yaw_pitch_roll_iter: Iterable
    hmd_dataset = None

    @property
    def yaw_pitch_roll_by_frame(self) -> list:
        chunk = int(self.chunk) - 1
        start = chunk * 30
        return self.user_hmd_data[slice(start, start + 30)]

    @property
    def seen_tiles(self):
        keys = [self.name, self.projection, self.tiling, self.user]
        return get_nested_value(self.get_tiles, keys)['chunks'][self.chunk]

    @property
    def user_hmd_data(self):
        return self.hmd_dataset[self.name + '_nas'][self.user]


class ViewportQuality(Props):
    def init(self):
        self.hmd_dataset = load_json(self.config.dataset_file)
        self.get_tiles = load_json(self.seen_tiles_result_pickle)

        self.proj_obj = {}
        for self.tiling in self.tiling_list:
            self.proj_obj[self.tiling] = build_projection(proj_name=self.projection,
                                                          tiling=self.tiling,
                                                          proj_res=self.config.config_dict['scale'][self.projection],
                                                          vp_res='1320x1080',
                                                          fov_res=self.fov)

    def main(self):
        """
        uvfq = User Viewport Frame Quality
        :return:
        """
        for _ in self.iter_name_user_tiling_chunk():
            if self.user_viewport_quality_json.exists():
                continue

            self.results = defaultdict(list)
            for self.quality in self.quality_list:
                self.worker()
            save_json(self.results, self.user_viewport_quality_json)
        self.collect_result()

    def iter_name_user_tiling_chunk(self):
        for self.name in self.name_list:
            for self.user in self.users_list:
                for self.tiling in self.tiling_list:
                    for self.chunk in self.chunk_list:
                        yield

    def worker(self):
        tile_ref_frame_reader = self.make_ref_vreader()
        tile_deg_frame_reader = self.make_deg_vreader()
        self.ui = ProgressBar(total=30, desc=self.__class__.__name__)

        for self.frame in range(30):
            self.ui.update(f'{self}')
            _mse, _ssim = self.calc_error(tile_deg_frame_reader,
                                          tile_ref_frame_reader)
            self.results['mse'].append(_mse)
            self.results['ssim'].append(_ssim)

        self.frame = None

    def collect_result(self):
        total = len(self.tiling_list) * 30 * len(self.quality_list) * 60
        for self.name in self.name_list:
            ui = ProgressBar(total=total, desc=self.__class__.__name__)
            if self.user_viewport_quality_result_json.exists(): continue

            result = AutoDict()
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    for self.quality in self.quality_list:
                        for self.chunk in self.chunk_list:
                            ui.update(f'{self}')
                            chunk_results = load_json(self.user_viewport_quality_json)
                            keys = (self.name, self.projection, self.tiling, self.user, self.quality, self.chunk)
                            set_nested_value(result, keys, chunk_results)

            save_json(result, self.user_viewport_quality_result_json)

    def calc_error(self, tile_deg_frame_reader, tile_ref_frame_reader):
        frame_proj_ref = tile_ref_frame_reader.get_frame()
        frame_proj_deg = tile_deg_frame_reader.get_frame()
        yaw_pitch_roll = self.yaw_pitch_roll_by_frame[self.frame]
        viewport_frame = self.get_vp_frame(frame_proj_deg, frame_proj_ref,
                                           yaw_pitch_roll)
        _mse = mse(*viewport_frame)
        _ssim = ssim(*viewport_frame, data_range=255.0, gaussian_weights=True,
                     sigma=1.5, use_sample_covariance=False)
        return _mse, _ssim

    def get_vp_frame(self, frame_proj_deg, frame_proj_ref, yaw_pitch_roll):
        viewport_frame_ref = self.proj_obj.extract_viewport(frame_proj_ref, yaw_pitch_roll)  # .astype('float64')
        viewport_frame_deg = self.proj_obj.extract_viewport(frame_proj_deg, yaw_pitch_roll)  # .astype('float64')
        return viewport_frame_deg, viewport_frame_ref

    def make_ref_vreader(self):
        tiles_path = {self.tile: self.reference_chunk
                      for self.tile in self.seen_tiles}
        tile_frame_reader = MountFrame(tiles_path, self.ctx)
        return tile_frame_reader

    def make_deg_vreader(self):
        tiles_path = {self.tile: self.decodable_chunk
                      for self.tile in self.seen_tiles}
        tile_frame_reader = MountFrame(tiles_path, self.ctx)
        return tile_frame_reader
