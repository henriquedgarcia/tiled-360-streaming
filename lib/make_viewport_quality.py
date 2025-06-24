import json
import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Optional, Iterator

import numpy as np
import pandas as pd
from pandas import DataFrame
from py360tools import CMP, ERP, ProjectionBase
from py360tools.utils import LazyProperty
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.assets.chunkprojectionreader import ChunkProjectionReader
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import save_json, load_json, print_error, idx2xy, iter_video

Tiling = str
Tile = str
NumpyArray = np.ndarray


class Props(Worker, ViewportQualityPaths, ABC):
    seen_tiles_deg_path: dict[Tile, Path]
    seen_tiles_ref_path: dict[Tile, Path]
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

    @LazyProperty
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

    @property
    def projection(self):
        return self.ctx.projection

    @projection.setter
    def projection(self, value):
        self.ctx.projection = value
        self.seen_tiles_db = pd.read_pickle(self.seen_tiles_result)

    def init(self):
        self.seen_tiles_level = ['name', 'projection', 'tiling', 'user', 'chunk']

    def get_seen_tiles(self) -> list[int]:
        seen_tiles = self.seen_tiles_db.xs((self.name, self.projection, self.tiling, int(self.user), int(self.chunk) - 1),
                                           level=self.seen_tiles_level)
        a = set()
        for item in list(seen_tiles['tiles_seen']):
            a.update(item)
        return list(a)

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
        for _ in self.iterate_name_projection_tiling:
            self.make_proj_obj()

            for _ in self.iterate_user_chunks:
                self.viewport_frame_ref_3dArray = None

                for self.quality in self.quality_list:
                    if (self.user_viewport_quality_json.exists()
                            and self.user_viewport_quality_json.stat().st_size > 10):
                        # print_error(f'{self.ctx}. File exists. skipping.')
                        continue

                    print(f'{self}. Processing...')
                    try:
                        if self.viewport_frame_ref_3dArray is None:
                            self.make_viewport_frame_ref_3dArray()
                    except StopIteration:
                        print_error(f'{self}. Decode error. Frame {self.frame}.')
                        self.logger.register_log(f'Decode error. Frame {self.frame}', self.user_viewport_quality_json)
                        continue

                    try:
                        self.calc_chunk_error_per_frame()
                        save_json(self.results, self.user_viewport_quality_json)
                    except StopIteration:
                        print_error(f'{self}. Decode error. Frame {self.frame}.')
                        self.logger.register_log(f'Decode error. Frame {self.frame}', self.user_viewport_quality_json)
                        continue

    def make_proj_obj(self):
        proj_obj = CMP if self.projection == 'cmp' else ERP
        self.proj_obj = p(tiling=self.tiling, proj_res=self.scale,
                          vp_res='1080x1080', fov_res='90x90')

    def make_viewport_frame_ref_3dArray(self):
        """
        Extrai todos os viewport de referência deste usuário para um chunk.
        :return:
        """
        tiles_reader: dict[str, Iterator]
        tiles_seen: dict[str, Path]
        tile_positions: dict[str, tuple[int, int, int, int]]

        tiles_seen = {str(self.tile): self.reference_chunk
                      for self.tile in self.get_seen_tiles()}
        tile_positions = make_tile_positions(self.proj_obj)
        tiles_reader = {tile: iter_video(file_path, gray=True)
                        for tile, file_path in tiles_seen.items()}
        proj_canvas = np.zeros(self.video_shape, dtype='uint8')
        viewport_30frames_ref = np.zeros((30, 1080, 1320))

        for frame, yaw_pitch_roll in enumerate(self.chunk_yaw_pitch_roll_per_frame):
            for tile in tiles_seen:
                x_ini, x_end, y_ini, y_end = tile_positions[tile]
                try:
                    tile_frame = next(tiles_reader[tile])
                except StopIteration:
                    msg = f'{self}. Decode error. {frame=}, {tile=}.'
                    raise AbortError(msg)
                proj_canvas[y_ini:y_end, x_ini:x_end] = tile_frame
            viewport_30frames_ref[frame] = proj_canvas

    def calc_chunk_error_per_frame(self, ):
        self.results = defaultdict(list)

        deg_tiles_path = {self.tile: self.decodable_chunk for self.tile in self.get_seen_tiles()}
        deg_proj_frame_vreader = ChunkProjectionReader(deg_tiles_path, proj=self.proj_obj)

        for self.frame, yaw_pitch_roll in enumerate(self.chunk_yaw_pitch_roll_per_frame):
            viewport_frame_deg = deg_proj_frame_vreader.extract_viewport(yaw_pitch_roll)
            _mse = mse(self.viewport_frame_ref_3dArray[self.frame], viewport_frame_deg)
            _ssim = ssim(self.viewport_frame_ref_3dArray[self.frame], viewport_frame_deg, data_range=255.0,
                         gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            self.results['mse'].append(_mse)
            self.results['ssim'].append(_ssim)
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


def make_tile_positions(proj: ProjectionBase) -> dict[str, tuple[int, int, int, int]]:
    """
    Um dicionário do tipo {tile: (x_ini, x_end, y_ini, y_end)}
    onde tiles é XXXX (verificar)
    e as coordenadas são inteiros.

    Mostra a posição inicial e final do tile na projeção.
    :param proj:
    :return:
    """
    tile_positions = {}
    tile_h, tile_w = proj.tiling.tile_shape
    tile_N, tile_M = proj.tiling.shape

    tile_list = list(map(int, proj.tiling.tile_list))

    for tile in tile_list:
        tile_m, tile_n = idx2xy(tile, (tile_N, tile_M))
        tile_y, tile_x = tile_n * tile_h, tile_m * tile_w
        x_ini = tile_x
        x_end = tile_x + tile_w
        y_ini = tile_y
        y_end = tile_y + tile_h
        tile_positions[str(tile)] = x_ini, x_end, y_ini, y_end
    return tile_positions


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

    config_file = Path('config/config_cmp_crf.json')
    # config_file = Path('config/config_cmp_qp.json')
    # config_file = Path('config/config_erp_qp.json')
    videos_file = Path('config/videos_reduced.json')
    # videos_file = Path('config/videos_full.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    ViewportQuality(ctx)
    # CheckViewportQuality(ctx)
