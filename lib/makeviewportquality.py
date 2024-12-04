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

    user_viewport_quality_dict_byframe: dict

    proj_obj: ProjectionBase
    yaw_pitch_roll_iter: Iterator

    get_tiles: dict
    seen_tiles: list[str]
    seen_tiles_deg: dict[str, Path]
    seen_tiles_ref: dict[str, Path]
    canvas: np.ndarray
    ui: ProgressBar

    def main(self):
        self.init()

        for _ in self.iter_name_proj_tiling_user_qlt_chunk():
            with self.task():
                self.worker()

    def init(self):
        self.viewport_quality_paths = ViewportQualityPaths(self.ctx)

    def iter_name_proj_tiling_user_qlt_chunk(self):
        for self.name in self.name_list:
            self.start_ui()
            self.load_get_tiles()

            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:

                    self.create_projections()

                    for self.user in self.users_list:
                        for self.quality in self.quality_list:
                            self.yaw_pitch_roll_iter = iter(self.user_hmd_data)

                            for self.chunk in self.chunk_list:
                                yield
            del self.ui

    def start_ui(self):
        print(f'==== {self.__class__.__name__} {self.ctx} ====')
        total = (len(self.name_list) * len(self.projection_list) * len(self.tiling_list)
                 * len(self.users_list) * len(self.quality_list) * len(self.chunk_list))
        self.ui: ProgressBar = ProgressBar(total=total, desc=f'{self.__class__.__name__}')

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
        self.tile = None
        self.ui.update(f'{self.ctx}')
        self.user_viewport_quality_dict_byframe = defaultdict(list)
        try:
            yield
        except AbortError as e:
            print_error(f'\t{e.args[0]}')
            return

        save_json(self.user_viewport_quality_dict_byframe, self.viewport_quality_paths.user_viewport_quality_json)

    def check_viewport_quality(self):
        try:
            size = self.viewport_quality_paths.user_viewport_quality_json.stat().st_size
            if size == 0:
                self.viewport_quality_paths.user_viewport_quality_json.unlink()
                raise FileNotFoundError()
            raise AbortError(f'The user_viewport_quality_json exist. Skipping')
        except FileNotFoundError:
            pass

    def worker(self):
        self.check_viewport_quality()
        self.update_seen_tiles()

        tile_ref_frame_reader = MountFrame(self.seen_tiles_ref, self.ctx)
        tile_deg_frame_reader = MountFrame(self.seen_tiles_deg, self.ctx)

        # para cada frame do chunk
        for self.frame in range(30):
            self.ui.set_postfix_str(f'{self.ctx}')  # 30 frames per chunk
            frame_proj_ref = tile_ref_frame_reader.get_frame()
            frame_proj_deg = tile_deg_frame_reader.get_frame()

            yaw_pitch_roll = next(self.yaw_pitch_roll_iter)
            viewport_frame_ref = self.proj_obj.extract_viewport(frame_proj_ref, yaw_pitch_roll)  # .astype('float64')
            viewport_frame_deg = self.proj_obj.extract_viewport(frame_proj_deg, yaw_pitch_roll)  # .astype('float64')

            _mse = mse(viewport_frame_ref, viewport_frame_deg)
            _ssim = ssim(viewport_frame_ref, viewport_frame_deg,
                         data_range=255.0, gaussian_weights=True, sigma=1.5,
                         use_sample_covariance=False, channel_axis=2)

            self.user_viewport_quality_dict_byframe['mse'].append(_mse)
            self.user_viewport_quality_dict_byframe['ssim'].append(_ssim)

    def update_seen_tiles(self):
        keys = [self.name, self.projection, self.tiling, self.user]
        self.seen_tiles = get_nested_value(self.get_tiles, keys)['chunks'][self.chunk]
        self.seen_tiles_ref = {self.tile: self.viewport_quality_paths.reference_chunk
                               for self.tile in self.seen_tiles}
        self.seen_tiles_deg = {self.tile: self.viewport_quality_paths.decodable_chunk
                               for self.tile in self.seen_tiles}
