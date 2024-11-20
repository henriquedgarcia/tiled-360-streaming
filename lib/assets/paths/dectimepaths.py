from pathlib import Path

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.ctxinterface import CtxInterface


class DectimePaths(CtxInterface):
    def __init__(self, context: Context):
        self.ctx = context
        self.base_paths = BasePaths(context)
        self.decodable_paths = MakeDecodablePaths(context)

    @property
    def dectime_folder(self) -> Path:
        return self.base_paths.dectime_folder / self.base_paths.basename_lv5

    @property
    def dectime_log(self) -> Path:
        chunk = int(self.ctx.chunk)
        return self.dectime_folder / f'chunk{chunk:03d}_dectime.log'

    @property
    def decodable_chunk(self) -> Path:
        return self.decodable_paths.decodable_chunk

    @property
    def dectime_result_json(self) -> Path:
        return self.base_paths.results_folder / f'dectime/time_{self.ctx.name}.json'
