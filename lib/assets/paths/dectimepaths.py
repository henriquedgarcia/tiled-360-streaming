from pathlib import Path

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths


class DectimePaths(BasePaths):
    def __init__(self, context: Context):
        self.ctx = context
        self.decodable_paths = MakeDecodablePaths(context)

    @property
    def dectime_folder(self) -> Path:
        return self.dectime_folder / self.folder_name_proj_tiling_tile_qlt

    @property
    def dectime_log(self) -> Path:
        chunk = int(self.ctx.chunk)
        return self.dectime_folder / f'chunk{chunk:03d}_dectime.log'

    @property
    def decodable_chunk(self) -> Path:
        return self.decodable_paths.decodable_chunk

    @property
    def dectime_result_json(self) -> Path:
        return self.results_folder / f'dectime/time_{self.ctx.name}.json'

    @property
    def dectime_result_pickle(self) -> Path:
        return self.results_folder / f'dectime/dectime_{self.name}.pickle'
