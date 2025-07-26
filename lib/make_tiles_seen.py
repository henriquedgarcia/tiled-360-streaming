import os
from pathlib import Path
from typing import Optional, Any

from py360tools import ProjectionBase, Viewport, ERP, CMP

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.paths.make_tiles_seen_paths import TilesSeenPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task, timer
from lib.utils.util import save_json, build_projection, get_nested_value


class PrivatesMethods(TilesSeenPaths):
    _results: dict = None

    @property
    def results(self) -> Optional[Any]:
        keys = [self.name, self.projection, self.tiling, self.user]
        try:
            value = get_nested_value(self._results, keys)
        except TypeError:
            value = None
        return value

    @results.setter
    def results(self, value: Any):
        keys = [self.name, self.projection, self.tiling]
        new_item = {self.user: value}
        try:
            get_nested_value(self._results, keys).update(new_item)
        except TypeError:
            self._results = AutoDict()
            get_nested_value(self._results, keys).update(new_item)

    viewport_dict: dict

    def get_tiles_seen_by_frame(self, user_hmd_data) -> list[list[str]]:
        if self.tiling == '1x1':
            return [["0"]] * self.n_frames

        tiles_seen_by_frame = []
        viewport_obj = self.viewport_dict[self.projection][self.tiling]

        for frame, yaw_pitch_roll in enumerate(user_hmd_data, 1):
            print(f'\r\tframe {frame:04d}/{self.n_frames}', end='')
            vptiles = viewport_obj.get_vptiles(yaw_pitch_roll)
            vptiles: list[str] = list(map(str, map(int, vptiles)))
            tiles_seen_by_frame.append(vptiles)
        return tiles_seen_by_frame

    def get_tiles_seen_by_chunk(self, tiles_seen_by_frame: list[list[str]]) -> dict[str, list[str]]:
        tiles_seen_by_chunk = {}

        if self.tiling == '1x1':
            duration = int(self.config.duration)
            return {str(i): ["0"] for i in range(1, duration + 1)}

        tiles_in_chunk = set()
        for frame, vptiles in enumerate(tiles_seen_by_frame):
            tiles_in_chunk.update(vptiles)

            if (frame + 1) % 30 == 0:
                chunk_id = frame // 30 + 1  # chunk start from 1
                tiles_seen_by_chunk[f'{chunk_id}'] = list(tiles_in_chunk)
                tiles_in_chunk.clear()
        return tiles_seen_by_chunk


class MakeTilesSeen(Worker, PrivatesMethods):
    viewport_dict: dict['str', dict['str', ProjectionBase]]
    tiles_seen: dict

    def init(self):
        def create_projections_dict():
            """
            viewport_dict = {'cmp': {'3x2': CMP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res),
                                       '6x4': CMP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res),
                                       ...},
                               'erp': {'3x2': ERP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res),
                                       '6x4': ERP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res),
                                       ...}
                               }
            :return:
            """
            projection_dict = AutoDict()
            for tiling in self.tiling_list:
                projection = ERP(tiling=tiling, proj_res=self.config.config_dict['scale']['erp'])
                vp = Viewport('1320x1080', self.fov, projection)
                projection_dict['erp'][tiling] = vp

                projection = CMP(tiling=tiling, proj_res=self.config.config_dict['scale']['cmp'])
                vp = Viewport('1320x1080', self.fov, projection)
                projection_dict['cmp'][tiling] = vp
            return projection_dict

        self.projection_dict = create_projections_dict()

    def main(self):
        for _ in self.iterate_name_projection_tiling_user():
            with task(self):
                self.check_seen_tiles()
                self.check_user_hmd_data()
                self.get_tiles_seen()
                self.save_tiles_seen()

    def check_seen_tiles(self):
        try:
            size = self.user_seen_tiles_json.stat().st_size
        except FileNotFoundError:
            return

        if size < 10:
            self.user_seen_tiles_json.unlink(missing_ok=True)
            return

        raise AbortError('Seen tiles JSON is OK.')

    def check_user_hmd_data(self):
        if self.user_hmd_data == {}:
            self.logger.register_log(f'HMD samples is missing, '
                                     f'user{self.user}',
                                     self.config.dataset_file)
            raise AbortError(f'HMD samples is missing, '
                             f'user{self.user}')

    def get_tiles_seen(self):
        print('')
        with timer(ident=1):
            tiles_seen_by_frame = self.get_tiles_seen_by_frame(self.user_hmd_data)
            # tiles_seen_by_chunk = self.get_tiles_seen_by_chunk(tiles_seen_by_frame)

            self.tiles_seen = {'frames': tiles_seen_by_frame,
                               # 'chunks': tiles_seen_by_chunk
                               }

    def save_tiles_seen(self):
        save_json(self.tiles_seen, self.user_seen_tiles_json)

    _results: dict


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
    # config_file = Path('config/config_erp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    MakeTilesSeen(ctx)
