from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths


class GetTilesPaths:
    def __init__(self, config: Config, context: Context):
        self.config = config
        self.ctx = context
        self.base_paths = BasePaths(config, context)

    @property
    def get_tiles_folder(self) -> Path:
        return self.base_paths.project_path / 'get_tiles'

    @property
    def get_tiles_json(self) -> Path:
        filename = (f'get_tiles_{self.config.dataset_file.stem}_{self.ctx.projection}_{self.ctx.name}'
                    f'_fov{self.config.fov}.json')
        return self.get_tiles_folder / filename

    @property
    def user_tiles_seen_json(self) -> Path:
        folder = self.get_tiles_folder / self.ctx.name / self.ctx.projection / self.ctx.tiling
        filename = f'seen_tiles_user{self.ctx.user}.json'
        return folder / filename

    @property
    def counter_tiles_json(self):
        filename = (f'counter_{self.config.dataset_file.stem}_{self.ctx.projection}_{self.ctx.name}'
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
