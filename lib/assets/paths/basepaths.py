from pathlib import Path

from config.config import Config
from lib.assets.ctxinterface import CtxInterface


class BasePaths(CtxInterface):
    config: Config

    @property
    def project_path(self):
        return self.config.project_folder

    @property
    def results_folder(self):
        return self.project_path / 'results_json'

    @property
    def lossless_folder(self):
        return self.project_path / 'lossless'

    @property
    def dash_folder(self):
        return self.project_path / 'dash'

    @property
    def decodable_folder0(self):
        return self.project_path / 'decodable'

    @property
    def dectime_folder(self):
        return self.project_path / 'dectime'

    @property
    def tiles_folder(self):
        return self.project_path / 'tiles'

    @property
    def quality_folder(self):
        return self.project_path / 'quality'

    @property
    def user_quality_folder(self):
        return self.project_path / 'user_quality'

    @property
    def dataset_folder(self) -> Path:
        return Path('datasets')

    @property
    def graphs_folder(self):
        return self.project_path / 'graphs'

    @property
    def viewport_quality_folder(self):
        return self.project_path / 'viewport_quality'

    @property
    def siti_folder(self):
        return self.project_path / 'siti'

    @property
    def seen_tiles_folder(self):
        return self.project_path / 'get_tiles'

    @property
    def folder_name(self):
        return Path(f'{self.name}')

    @property
    def folder_name_proj(self):
        return self.folder_name / Path(f'{self.projection}')

    @property
    def folder_name_proj_tiling(self):
        return self.folder_name_proj / Path(f'{self.tiling}')

    @property
    def folder_name_proj_tiling_tile(self):
        return self.folder_name_proj_tiling / Path(f'tile{self.tile}')

    @property
    def folder_name_proj_tiling_tile_qlt(self):
        return self.folder_name_proj_tiling_tile / Path(f'{self.config.rate_control}{self.quality}')
