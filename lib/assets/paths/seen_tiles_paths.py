from pathlib import Path

from lib.assets.paths.basepaths import BasePaths


class SeenTilesPaths(BasePaths):
    @property
    def seen_tiles_result_json(self) -> Path:
        """depend on name and fov"""
        return self.seen_tiles_folder / f'seen_tiles_{self.name}_fov{self.fov}.json'

    @property
    def seen_tiles_result_pickle(self) -> Path:
        """depend on name and fov"""
        return self.seen_tiles_folder / f'seen_tiles.pickle'

    @property
    def user_seen_tiles_folder(self) -> Path:
        return self.seen_tiles_folder / self.folder_name_proj_tiling

    @property
    def user_seen_tiles_json(self) -> Path:
        return self.user_seen_tiles_folder / f'user{int(self.user):02d}.json'

    @property
    def counter_tiles_json(self):
        filename = (f'counter_{self.name}_{self.projection}'
                    f'_fov{self.fov}.json')
        folder = self.seen_tiles_folder / 'counter'
        return folder / filename

    @property
    def heatmap_tiling(self):
        filename = (f'heatmap_tiling_{self.dataset_name}_{self.projection}_{self.name}_{self.tiling}'
                    f'_fov{self.fov}.png')
        folder = self.seen_tiles_folder / 'heatmap'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / filename

    @property
    def video_test_folder(self):
        folder = self.seen_tiles_folder / 'video_test'
        folder.mkdir(parents=True, exist_ok=True)
        return folder
