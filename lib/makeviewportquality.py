from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from py360tools import ProjectionBase
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.mountframe import MountFrame
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.worker import Worker, ProgressBar
from lib.utils.util import build_projection, print_error, save_json, load_json, get_nested_value


class ViewportQuality(Worker, CtxInterface):
    viewport_quality_paths: ViewportQualityPaths

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
    yaw_pitch_roll_iter: Iterator

    def main(self):
        self.init()
        self.main_loop()

    def main_loop(self):
        for self.name in self.name_list:
            self.load_get_tiles()

            for self.tiling in self.tiling_list:
                if self.tiling == '1x1': continue
                self.create_projections()

                for self.user in self.users_list:
                    self.yaw_pitch_roll_iter = iter(self.user_hmd_data)

                    for self.chunk in self.chunk_list:
                        self.update_seen_tiles()
                        self.yaw_pitch_roll_by_frame = [next(self.yaw_pitch_roll_iter)
                                                        for self.frame in range(30)]

                        for self.quality in self.quality_list:
                            with self.task():
                                self.worker()

    def worker(self):
        self.check_viewport_quality()
        self.start_ui(30)

        tile_ref_frame_reader = self.make_ref_vreader()
        tile_deg_frame_reader = self.make_deg_vreader()

        for self.frame in range(30):
            self.ui.update(f'frame{self.frame:02d}')

            _mse, _ssim = self.calc_error(tile_deg_frame_reader,
                                          tile_ref_frame_reader)

            self.results['mse'].append(_mse)
            self.results['ssim'].append(_ssim)

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
        tiles_path = {self.tile: self.viewport_quality_paths.reference_chunk
                      for self.tile in self.seen_tiles}
        tile_frame_reader = MountFrame(tiles_path, self.ctx)
        return tile_frame_reader

    def make_deg_vreader(self):
        tiles_path = {self.tile: self.viewport_quality_paths.decodable_chunk
                      for self.tile in self.seen_tiles}
        tile_frame_reader = MountFrame(tiles_path, self.ctx)
        return tile_frame_reader

    def init(self):
        self.viewport_quality_paths = ViewportQualityPaths(self.ctx)
        self.projection = 'cmp'

    def start_ui(self, total):
        self.ui = ProgressBar(total=total, desc=f'{self.__class__.__name__}: '
                                                f'{self.name}_{self.tiling}_'
                                                f'{self.user}_{self.chunk}_'
                                                f'{self.quality}_')

    def load_get_tiles(self):
        self.get_tiles: dict = load_json(self.viewport_quality_paths.get_tiles_result_json)

    def create_projections(self):
        self.proj_obj = build_projection(proj_name=self.projection,
                                         tiling=self.tiling,
                                         proj_res=self.config.config_dict['scale'][self.projection],
                                         vp_res='1320x1080',
                                         fov_res=self.fov)

    @contextmanager
    def task(self):
        try:
            self.results = defaultdict(list)
            yield
        except AbortError as e:
            print_error(f'\t{e.args[0]}')
            return

        save_json(self.results,
                  self.viewport_quality_paths.user_viewport_quality_json)

    def check_viewport_quality(self):
        try:
            size = self.viewport_quality_paths.user_viewport_quality_json.stat().st_size
            if size == 0:
                self.viewport_quality_paths.user_viewport_quality_json.unlink()
                raise FileNotFoundError()
            raise AbortError(f'The user_viewport_quality_json exist. Skipping')
        except FileNotFoundError:
            pass

    def update_seen_tiles(self):
        keys = [self.name, self.projection, self.tiling, self.user]
        self.seen_tiles = get_nested_value(self.get_tiles, keys)['chunks'][self.chunk]
