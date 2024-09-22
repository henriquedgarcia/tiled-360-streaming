from pathlib import Path

from config.config import config
from lib.assets.context import ctx
from lib.assets.paths import base_paths


class SitiPaths:

    @property
    def siti_stats(self) -> Path:
        base_paths.siti_folder.mkdir(exist_ok=True, parents=True)
        return base_paths.siti_folder / f'siti_stats.csv'

    @property
    def siti_plot(self) -> Path:
        base_paths.siti_folder.mkdir(exist_ok=True, parents=True)
        return base_paths.siti_folder / f'siti_plot.png'

    @property
    def siti_results(self) -> Path:
        base_paths.siti_folder.mkdir(exist_ok=True, parents=True)
        name = f'siti_results'

        if ctx.name is not None:
            name += f'_{ctx.name}'
        if ctx.tiling is not None:
            name += f'_{ctx.tiling}'
        if ctx.quality is not None:
            name += f'_{config.rate_control}{ctx.quality}'
        if ctx.tile is not None:
            name += f'_tile{ctx.tile}'
        if ctx.chunk is not None:
            name += f'_chunk{ctx.chunk}'
        return base_paths.siti_folder / f'{name}.csv'


siti_paths = SitiPaths()
