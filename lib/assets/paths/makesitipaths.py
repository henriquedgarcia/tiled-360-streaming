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
    def siti_all_plot(self) -> Path:
        return self.siti_folder / f'siti_plot_all.png'

    @property
    def siti_name_plot(self) -> Path:
        folder = self.siti_folder / 'plot'
        folder.mkdir(exist_ok=True, parents=True)
        return folder / f'siti_plot_{self.name}.png'

    @property
    def siti_csv_results(self) -> Path:
        folder = self.siti_folder / 'siti_csv'
        folder.mkdir(exist_ok=True, parents=True)
        name = ''
        if self.name is not None:
            name += f'{self.name}'
        if self.tiling is not None:
            name += f'_{self.tiling}'
        if self.tile is not None:
            name += f'_tile{self.tile}'
        if self.quality is not None:
            name += f'{self.rate_control}{self.quality}'

        return folder / f"{name}.csv"
