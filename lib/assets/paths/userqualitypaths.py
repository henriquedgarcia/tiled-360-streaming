from pathlib import Path

import pandas as pd

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths


class UserQualityPaths:
    def __init__(self, context: Context):
        self.ctx = context
        self.config = context.config
        self.base_paths = BasePaths(context)

    @property
    def get_tiles_folder(self) -> Path:
        return self.base_paths.project_path / 'get_tiles'

    @property
    def get_tiles_json(self) -> Path:
        filename = (f'get_tiles_{self.ctx.name}_{self.ctx.projection}'
                    f'_fov{self.config.fov}.json')
        return self.get_tiles_folder / filename

    @property
    def user_tiles_seen_json(self) -> Path:
        folder = self.get_tiles_folder / self.ctx.name / self.ctx.projection / self.ctx.tiling
        filename = f'seen_tiles_user{self.ctx.user}.json'
        return folder / filename

    @property
    def counter_tiles_json(self):
        filename = (f'counter_{self.dataset_name}_{self.ctx.projection}_{self.ctx.name}'
                    f'_fov{self.config.fov}.json')
        folder = self.get_tiles_folder / 'counter'
        return folder / filename

    @property
    def heatmap_tiling(self):
        filename = (f'heatmap_tiling_{self.dataset_name}_{self.ctx.projection}_{self.ctx.name}_{self.ctx.tiling}'
                    f'_fov{self.config.fov}.png')
        folder = self.get_tiles_folder / 'heatmap'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / filename

    @property
    def video_test_folder(self):
        folder = self.get_tiles_folder / 'video_test'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def sph_file(self) -> Path:
        return self.config.sph_file

    @property
    def dataset_json(self) -> Path:
        return self.config.dataset_file

    @property
    def dataset_name(self):
        return self.config.dataset_file.stem

    _csv_dataset_file: Path

    @property
    def csv_dataset_file(self) -> Path:
        return self._csv_dataset_file

    head_movement: pd.DataFrame

    @csv_dataset_file.setter
    def csv_dataset_file(self, filename: Path) -> None:
        self._csv_dataset_file = filename
        self.head_movement = pd.read_csv(filename,
                                         names=['timestamp', 'Qx', 'Qy', 'Qz', 'Qw', 'Vx', 'Vy', 'Vz'])
