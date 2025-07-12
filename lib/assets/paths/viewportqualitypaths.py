from pathlib import Path

from lib.assets.paths.make_chunk_quality_paths import MakeChunkQualityPaths
from lib.assets.paths.make_tiles_seen_paths import TilesSeenPaths


class ViewportQualityPaths(TilesSeenPaths, MakeChunkQualityPaths):
    @property
    def user_viewport_folder(self):
        folder = (self.viewport_quality_folder /
                  f'{self.name}' / f'{self.projection}' / f'{self.tiling}' / f'{self.user}' / f'{self.quality}')
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
        return self.viewport_quality_folder / f'user_viewport_quality_{self.name}_{self.projection}_{self.rate_control}.pickle'

    @property
    def chunk_quality_result(self) -> Path:
        """depend on name and fov"""
        return self.results_folder / f'user_viewport_quality_{self.projection}_{self.rate_control}.pickle'
