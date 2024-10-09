from pathlib import Path

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths


class DectimePaths:
    def __init__(self, context: Context):
        self.config = context.config
        self.ctx = context
        self.base_paths = BasePaths(context)

    @property
    def dectime_folder(self) -> Path:
        return self.base_paths.dectime_folder / self.base_paths.basename2

    @property
    def dectime_log(self) -> Path:
        chunk = int(self.ctx.chunk)
        return self.dectime_folder / f'tile{self.ctx.tile}_{chunk:03d}_dectime.log'

    @property
    def dectime_result_json(self) -> Path:
        return self.base_paths.results_folder / f'time_{self.ctx.name}.json'
