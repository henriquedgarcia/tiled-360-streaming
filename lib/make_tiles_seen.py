import os
from pathlib import Path
from typing import Optional, Any

from py360tools import ProjectionBase, Viewport, ERP, CMP, Tile

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.paths.make_tiles_seen_paths import TilesSeenPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task, timer
from lib.utils.util import get_nested_value
from lib.utils.io_util import save_json


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

    def get_tiles_seen_by_frame(self) -> list[list[str]]:
        if self.tiling == '1x1': return [["0"]] * self.n_frames

        tiles_seen_by_frame = []
        viewport_obj: Viewport = self.viewport_dict[self.projection][self.tiling]

        for frame, yaw_pitch_roll in enumerate(self.user_hmd_data, 1):
            print(f'\r\tframe {frame:04d}/{self.n_frames}', end='')
            vptiles: list[Tile] = viewport_obj.get_vptiles(yaw_pitch_roll)
            vptiles: list[str] = [str(tile.idx) for tile in vptiles]
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
        proj_res = self.config.config_dict['scale']['erp']

        def create_viewport_dict():
            """
            viewport_dict = {'cmp': {'3x2': Viewport(vp_res, self.fov, CMP(tiling='3x2', proj_res=proj_res)),
                                     '6x4': Viewport(vp_res, self.fov, CMP(tiling='6x4', proj_res=proj_res)),
                                     ...},
                             'erp': {'3x2': Viewport(vp_res, self.fov, ERP(tiling='3x2', proj_res=proj_res)),
                                     '6x4': Viewport(vp_res, self.fov, ERP(tiling='6x4', proj_res=proj_res)),
                                     ...}
                             }
            :return:
            """
            viewport_dict = AutoDict()
            for tiling in self.tiling_list:
                viewport_dict['erp'][tiling] = Viewport(self.vp_res, self.fov, ERP(tiling=tiling, proj_res=proj_res))
                viewport_dict['cmp'][tiling] = Viewport(self.vp_res, self.fov, CMP(tiling=tiling, proj_res=proj_res))
            return viewport_dict

        self.viewport_dict = create_viewport_dict()

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
            tiles_seen_by_frame = self.get_tiles_seen_by_frame()
            self.tiles_seen = {'frames': tiles_seen_by_frame}

    def save_tiles_seen(self):
        save_json(self.tiles_seen, self.user_seen_tiles_json)

    _results: dict


if __name__ == '__main__':
    os.chdir('../')

    # config_file = Path('config/config_cmp_qp.json')
    # videos_file = Path('config/videos_reduced.json')

    config_file = Path('config/config_pres_qp.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    MakeTilesSeen(ctx).run()
