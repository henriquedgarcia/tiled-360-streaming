from pathlib import Path

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths


class GetTilesPaths(CtxInterface):
    ctx: Context

    def __init__(self, context: Context):
        self.ctx = context
        self.base_paths = BasePaths(context)

    @property
    def get_tiles_result_json(self) -> Path:
        return self.base_paths.results_folder / f'get_tiles_{self.name}_{self.projection}_fov{self.fov}.json'

    @property
    def user_tiles_seen_folder(self) -> Path:
        return self.base_paths.get_tiles_folder / self.base_paths.basename_lv3

    @property
    def user_tiles_seen_json(self) -> Path:
        return self.user_tiles_seen_folder / f'user{int(self.user):02d}.json'

    @property
    def counter_tiles_json(self):
        filename = (f'counter_{self.name}_{self.projection}'
                    f'_fov{self.fov}.json')
        folder = self.base_paths.get_tiles_folder / 'counter'
        return folder / filename

    @property
    def heatmap_tiling(self):
        filename = (f'heatmap_tiling_{self.dataset_name}_{self.projection}_{self.name}_{self.tiling}'
                    f'_fov{self.fov}.png')
        folder = self.base_paths.get_tiles_folder / 'heatmap'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / filename

    @property
    def video_test_folder(self):
        folder = self.base_paths.get_tiles_folder / 'video_test'
        folder.mkdir(parents=True, exist_ok=True)
        return folder
