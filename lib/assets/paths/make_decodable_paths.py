from pathlib import Path

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.makedashpaths import MakeDashPaths


class MakeDecodablePaths(CtxInterface):
    def __init__(self, context: Context):
        self.ctx = context
        self.base_paths = BasePaths(context)
        self.make_dash_paths = MakeDashPaths(context)

    @property
    def decodable_folder(self) -> Path:
        return self.base_paths.decodable_folder / self.base_paths.basename_lv5

    @property
    def decodable_chunk(self) -> Path:
        return self.decodable_folder / f'chunk{self.chunk}.mp4'

    def mpd_folder(self) -> Path:
        return self.make_dash_paths.mpd_folder

    @property
    def dash_m4s(self) -> Path:
        return self.make_dash_paths.dash_m4s

    @property
    def dash_init(self) -> Path:
        return self.make_dash_paths.dash_init

    @property
    def bitrate_result_json(self) -> Path:
        return self.make_dash_paths.bitrate_result_json
