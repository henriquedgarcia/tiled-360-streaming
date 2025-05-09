from pathlib import Path

from lib.assets.paths.basepaths import BasePaths


class ViewportQualityPaths(BasePaths):
    @property
    def user_viewport_quality_json(self) -> Path:
        """
        Need name tiling user quality chunk
        """
        return (self.viewport_quality_folder /
                f'{self.name}' / f'{self.tiling}' / f'{self.user}' / f'{self.quality}' /
                f'user_viewport_quality_chunk{self.chunk}.json')

    @property
    def user_viewport_quality_result_json(self) -> Path:
        """
        Need name tiling user quality chunk
        """

        return self.results_folder / f'user_viewport_quality/user_viewport_quality_{self.name}.json'

    @property
    def user_viewport_quality_result_pickle(self) -> Path:
        """
        Need name tiling user quality chunk
        """
        folder = self.results_folder / f'user_viewport_quality'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'user_viewport_quality_{self.name}.pickle'
