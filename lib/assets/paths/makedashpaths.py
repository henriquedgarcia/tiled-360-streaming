from pathlib import Path

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.maketilespaths import MakeTilesPaths


class MakeDashPaths(CtxInterface):
    def __init__(self, context: Context):
        self.ctx = context
        self.base_paths = BasePaths(context)
        self.make_tiles_paths = MakeTilesPaths(context)

    @property
    def tile_video(self) -> Path:
        return self.make_tiles_paths.tile_video

    @property
    def mp4box_log(self) -> Path:
        return self.base_paths.dash_folder / self.base_paths.basename_lv3 / f'tile{self.tile}_segmenter.log'

    @property
    def mpd_folder(self) -> Path:
        return self.base_paths.dash_folder / self.base_paths.basename_lv4

    @property
    def dash_mpd(self) -> Path:
        return self.mpd_folder / f'tile{self.tile}.mpd'

    @property
    def dash_m4s(self) -> Path:
        return self.mpd_folder / f'tile{self.tile}_{self.rate_control}{self.quality}_{self.chunk}.m4s'

    @property
    def dash_init(self) -> Path:
        return self.mpd_folder / f'tile{self.tile}_{self.rate_control}{self.quality}_init.mp4'

    @property
    def bitrate_result_json(self) -> Path:
        return self.base_paths.results_folder / f'bitrate/{self.name}.json'
