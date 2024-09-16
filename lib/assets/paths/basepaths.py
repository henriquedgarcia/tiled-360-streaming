from pathlib import Path

from config.config import Config
from lib.assets.context import Context


class BasePaths:
    def __init__(self, config: Config, ctx: Context):
        self.config = config
        self.ctx = ctx

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
    def segmenter_folder(self):
        return self.project_path / 'chunks'

    @property
    def tiles_folder(self):
        return self.project_path / 'tiles'

    @property
    def quality_folder(self):
        return self.project_path / 'quality'

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
    def basename1(self):
        return (Path(f'{self.ctx.name}') /
                f'{self.ctx.projection}' /
                f'{self.config.rate_control}{self.ctx.quality}' /
                f'{self.ctx.tiling}'
                )

    @property
    def basename2(self):
        return self.basename1 / f'tile{self.ctx.tile}'
