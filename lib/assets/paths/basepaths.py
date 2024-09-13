from pathlib import Path

from config.config import config
from lib.assets.context import ctx


class BasePaths:
    @property
    def project_path(self):
        return config.project_folder

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
        return (Path(f'{ctx.name}') /
                f'{ctx.projection}' /
                f'{config.rate_control}{ctx.quality}' /
                f'{ctx.tiling}'
                )

    @property
    def basename2(self):
        return self.basename1 / f'tile{ctx.tile}'


base_paths = BasePaths()
