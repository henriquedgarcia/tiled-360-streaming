from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from py360tools import ProjectionBase
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from lib.assets.autodict import AutoDict
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
    yaw_pitch_roll_dict_iter: dict[str, Iterator]

    get_tiles: dict
    seen_tiles: list[str]
    seen_tiles_deg_path: dict[str, Path]
    seen_tiles_ref_path: dict[str, Path]
    canvas: np.ndarray
    ui: ProgressBar

    def main(self):
        self.init()
        self.iter_name_proj_tiling_user_qlt_chunk()

    def iter_name_proj_tiling_user_qlt_chunk(self):
        for self.name in self.name_list:
            self.load_get_tiles()

            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    if self.tiling == '1x1': continue
                    self.create_projections()
                    self.load_yaw_pitch_roll_iter()

                    for self.chunk in self.chunk_list:
                        with self.task():
                            self.worker()
                        self.tile = None

    def worker(self):
        file_check = AutoDict()
        for self.user in self.users_list:
            for self.quality in self.quality_list:
                try:
                    size = self.viewport_quality_paths.user_viewport_quality_json.stat().st_size
                    if size == 0:
                        self.viewport_quality_paths.user_viewport_quality_json.unlink()
                        raise FileNotFoundError()
                    file_check[self.quality][self.user] = True
                except FileNotFoundError:
                    file_check[self.quality][self.user] = False

        self.user_viewport_quality_dict_byframe = AutoDict()
        self.start_ui()

        tiles_ref_path = {self.tile: self.viewport_quality_paths.reference_chunk
                          for self.tile in self.tile_list}
        tile_ref_frame_reader = MountFrame(tiles_ref_path, self.ctx)

        tile_deg_frame_reader_dict: dict[str, MountFrame] = {}

        for self.quality in self.quality_list:
            tiles_deg_path = {self.tile: self.viewport_quality_paths.decodable_chunk
                              for self.tile in self.tile_list}
            tile_deg_frame_reader_dict[self.quality] = MountFrame(tiles_deg_path, self.ctx)

        self.hmd_sample_n = 0
        # para cada frame do chunk
        for self.frame in range(30):
            frame_proj_ref = tile_ref_frame_reader.get_frame()

            for self.quality in self.quality_list:
                frame_proj_deg = tile_deg_frame_reader_dict[self.quality].get_frame()

                for self.user in self.users_list:
                    if file_check[self.quality][self.user]: continue
                    self.ui.update(f'{self.frame}')

                    yaw_pitch_roll = self.user_hmd_data[self.hmd_sample_n]
                    viewport_frame_ref = self.proj_obj.extract_viewport(frame_proj_ref, yaw_pitch_roll)  # .astype('float64')
                    viewport_frame_deg = self.proj_obj.extract_viewport(frame_proj_deg, yaw_pitch_roll)  # .astype('float64')

                    _mse = mse(viewport_frame_ref, viewport_frame_deg)
                    _ssim = ssim(viewport_frame_ref, viewport_frame_deg,
                                 data_range=255.0, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)

                    try:
                        self.user_viewport_quality_dict_byframe[self.quality][self.user]['mse'].append(_mse)
                        self.user_viewport_quality_dict_byframe[self.quality][self.user]['ssim'].append(_ssim)
                    except AttributeError:
                        self.user_viewport_quality_dict_byframe[self.quality][self.user]['mse'] = [_mse]
                        self.user_viewport_quality_dict_byframe[self.quality][self.user]['ssim'] = [_ssim]

            self.hmd_sample_n += 1

        for self.user in self.users_list:
            for self.quality in self.quality_list:
                if file_check[self.quality][self.user]: continue
                save_json(self.user_viewport_quality_dict_byframe[self.quality][self.user], self.viewport_quality_paths.user_viewport_quality_json)

    hmd_sample_n: int

    def init(self):
        self.viewport_quality_paths = ViewportQualityPaths(self.ctx)

    def load_yaw_pitch_roll_iter(self):
        self.yaw_pitch_roll_dict_iter = {}
        for self.user in self.users_list:
            self.yaw_pitch_roll_dict_iter[self.user] = iter(self.user_hmd_data)

    def start_ui(self):
        print(f'==== {self.__class__.__name__} {self.ctx} ====')
        total = (len(self.users_list) * len(self.quality_list) * 30)
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
        try:
            yield
        except AbortError as e:
            print_error(f'\t{e.args[0]}')
            return

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
