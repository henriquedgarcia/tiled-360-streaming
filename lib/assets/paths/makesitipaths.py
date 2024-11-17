from pathlib import Path
from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.maketilespaths import MakeTilesPaths


class MakeSitiPaths(CtxInterface):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.base_paths = BasePaths(ctx)
        self.make_tiles_paths = MakeTilesPaths(ctx)

    @property
    def siti_folder(self) -> Path:
        self.base_paths.siti_folder.mkdir(exist_ok=True, parents=True)
        return self.base_paths.siti_folder

    @property
    def siti_stats(self) -> Path:
        return self.siti_folder / f'siti_stats.csv'

    @property
    def siti_plot(self) -> Path:
        return self.siti_folder / f'siti_plot.png'

    @property
    def siti_results(self) -> Path:
        name = f'siti_results'

        if self.name is not None:
            name += f'_{self.name}'
        if self.tiling is not None:
            name += f'_{self.tiling}'
        if self.quality is not None:
            name += f'_{self.rate_control}{self.quality}'
        if self.tile is not None:
            name += f'_tile{self.tile}'
        if self.chunk is not None:
            name += f'_chunk{self.chunk}'
        return self.siti_folder / f"siti/{self.name}.csv"
