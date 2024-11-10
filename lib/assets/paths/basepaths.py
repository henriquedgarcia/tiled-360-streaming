from pathlib import Path

from lib.assets.context import Context


class BasePaths:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.config = ctx.config

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
    def decodable_folder(self):
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
    def viewport_folder(self):
        return self.project_path / 'viewport'

    @property
    def siti_folder(self):
        return self.project_path / 'siti'

    @property
    def get_tiles_folder(self):
        return self.project_path / 'get_tiles'

    @property
    def basename_lv1(self):
        return Path(f'{self.ctx.name}')

    @property
    def basename_lv2(self):
        return self.basename_lv1 / Path(f'{self.ctx.projection}')

    @property
    def basename_lv3(self):
        return self.basename_lv2 / Path(f'{self.ctx.tiling}')

    @property
    def basename_lv4(self):
        return self.basename_lv3 / Path(f'tile{self.ctx.tile}')

    @property
    def basename_lv5(self):
        return self.basename_lv4 / Path(f'{self.config.rate_control}{self.ctx.quality}')
