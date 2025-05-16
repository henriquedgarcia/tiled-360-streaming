from pathlib import Path

from lib.assets.paths.basepaths import BasePaths


class ViewportQualityPaths(BasePaths):
    @property
    def user_viewport_folder(self):
        folder = (self.viewport_quality_folder /
                  f'{self.name}' / f'{self.tiling}' / f'{self.user}' / f'{self.quality}')
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def user_viewport_quality_json(self) -> Path:
        """
        Need name tiling user quality chunk
        """
        filename = f'user_viewport_quality_chunk{self.chunk}.json'
        return self.user_viewport_folder / filename

    @property
    def user_viewport_result_by_name(self) -> Path:
        return self.viewport_quality_folder / f'user_viewport_quality_{self.metric}_{self.name}_{self.projection}_{self.rate_control}.pickle'

    @property
    def chunk_quality_result(self) -> Path:
        """depend on name and fov"""
        return self.results_folder / f'chunk_quality_{self.metric}_{self.projection}_{self.rate_control}.pickle'
