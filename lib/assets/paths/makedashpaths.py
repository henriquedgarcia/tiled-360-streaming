from pathlib import Path

from lib.assets.paths.maketilespaths import MakeTilesPaths


class MakeDashPaths(MakeTilesPaths):
    @property
    def mp4box_log(self) -> Path:
        return self.dash_folder / self.folder_name_proj_tiling / f'tile{self.tile}_segmenter.log'

    @property
    def mpd_folder(self) -> Path:
        return self.dash_folder / self.folder_name_proj_tiling_tile

    @property
    def dash_mpd(self) -> Path:
        return self.mpd_folder / f'tile{self.tile}.mpd'

    @property
    def dash_init(self) -> Path:
        return self.mpd_folder / f'tile{self.tile}_{self.rate_control}{self.quality}_init.mp4'

    @property
    def dash_m4s(self) -> Path:
        return self.mpd_folder / f'tile{self.tile}_{self.rate_control}{self.quality}_{self.chunk}.m4s'

    @property
    def bitrate_result_json(self) -> Path:
        return self.results_folder / f'bitrate/bitrate_{self.name}.json'

    @property
    def bitrate_result_pickle(self) -> Path:
        return self.results_folder / f'bitrate/bitrate_{self.name}.pickle'
