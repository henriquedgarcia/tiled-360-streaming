from pathlib import Path

from lib.assets.paths.basepaths import BasePaths


class MakeSitiPaths(BasePaths):
    @property
    def siti_stats(self) -> Path:
        folder = self.siti_folder / 'siti_stats'
        folder.mkdir(exist_ok=True, parents=True)
        return folder / f'siti_stats.csv'

    @property
    def siti_result_pickle(self) -> Path:
        folder = self.results_folder
        return folder / f'siti.pickle'

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
