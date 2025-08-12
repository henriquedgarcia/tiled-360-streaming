from pathlib import Path

from lib.assets.paths.basepaths import BasePaths


class TilesSeenPaths(BasePaths):
    @property
    def seen_tiles_result_by_name(self) -> Path:
        """depend on name and fov"""
        'seen_tiles_cable_cam_cmp_fov110x90.pickle'
        return self.seen_tiles_folder / f'seen_tiles_{self.name}_{self.projection}_fov{self.fov}.pickle'

    @property
    def df_seen_tiles_hf5_by_name(self) -> Path:
        """depend on name and fov"""
        return self.seen_tiles_folder / f'df_seen_tiles_{self.name}_{self.projection}_fov{self.fov}.h5'

    @property
    def seen_tiles_result(self) -> Path:
        """depend on name and fov"""
        return self.results_folder / f'seen_tiles_fov{self.fov}.hd5'

    @property
    def user_seen_tiles_folder(self) -> Path:
        folder = self.seen_tiles_folder / self.folder_name_proj_tiling
        folder.mkdir(parents=True, exist_ok=True)
        return folder

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
