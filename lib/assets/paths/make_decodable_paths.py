from pathlib import Path

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.makedashpaths import MakeDashPaths


class MakeDecodablePaths(BasePaths):
    def __init__(self, context: Context):
        self.ctx = context
        self.make_dash_paths = MakeDashPaths(context)

    @property
    def decodable_folder(self) -> Path:
        return self.decodable_folder0 / self.folder_name_proj_tiling_tile_qlt

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


class IfDecodablePaths(CtxInterface):
    make_decodable_path: MakeDecodablePaths

    @property
    def decodable_folder(self):
        return self.make_decodable_path.decodable_folder

    @property
    def dash_init(self):
        return self.make_decodable_path.dash_init

    @property
    def dash_m4s(self):
        return self.make_decodable_path.dash_m4s

    @property
    def decodable_chunk(self):
        return self.make_decodable_path.decodable_chunk
