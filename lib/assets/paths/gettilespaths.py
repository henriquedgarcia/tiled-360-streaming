from pathlib import Path

import pandas as pd

from config.config import config
from lib.assets.context import ctx
from lib.assets.paths import base_paths


class GetTilesPaths:
    @property
    def get_tiles_folder(self) -> Path:
        return base_paths.project_path / 'get_tiles'

    @property
    def get_tiles_json(self) -> Path:
        filename = f'get_tiles_{config.dataset_file.stem}_{ctx.projection}_{ctx.name}_fov{config.fov}.json'
        return self.get_tiles_folder / filename

    @property
    def user_tiles_seen_json(self) -> Path:
        folder = self.get_tiles_folder / ctx.name / ctx.projection / ctx.tiling
        folder.mkdir(exist_ok=True, parents=True)
        filename = f'seen_tiles_user{ctx.user}.json'
        return folder / filename

    @property
    def counter_tiles_json(self):
        filename = f'counter_{config.dataset_file.stem}_{ctx.proj}_{ctx.name}_fov{ctx.fov}.json'
        folder = self.get_tiles_folder / 'counter'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / filename

    @property
    def heatmap_tiling(self):
        filename = f'heatmap_tiling_{self.dataset_name}_{ctx.proj}_{ctx.name}_{ctx.tiling}_fov{ctx.fov}.png'
        folder = self.get_tiles_folder / 'heatmap'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / filename

    @property
    def sph_file(self) -> Path:
        return config.sph_file

    @property
    def dataset_json(self) -> Path:
        return config.dataset_file

    @property
    def dataset_name(self):
        return config.dataset_file.stem

    _csv_dataset_file: Path

    @property
    def csv_dataset_file(self) -> Path:
        return self._csv_dataset_file

    @csv_dataset_file.setter
    def csv_dataset_file(self, value):
        self._csv_dataset_file = value
        user_nas_id, video_nas_id = self._csv_dataset_file.stem.split('_')
        ctx.video_name = ctx.video_id_map[video_nas_id]
        ctx.user_id = ctx.user_map[user_nas_id]

        names = ['timestamp', 'Qx', 'Qy', 'Qz', 'Qw', 'Vx', 'Vy', 'Vz']
        ctx.head_movement = pd.read_csv(self.csv_dataset_file, names=names)


get_tiles_paths = GetTilesPaths()
