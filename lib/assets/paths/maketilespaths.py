from pathlib import Path

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths


class MakeTilesPaths(CtxInterface):
    def __init__(self, context: Context):
        self.ctx = context
        self.base_paths = BasePaths(context)

    @property
    def lossless_video(self) -> Path:
        return self.base_paths.lossless_folder / self.projection / f'{self.name}.mp4'

    @property
    def lossless_log(self) -> Path:
        return self.lossless_video.with_suffix('.log')

    @property
    def tile_folder(self) -> Path:
        folder = self.base_paths.tiles_folder / self.base_paths.basename1
        return folder

    @property
    def tile_video(self) -> Path:
        return self.base_paths.tiles_folder / self.base_paths.basename1 / f'tile{self.tile}.mp4'

    @property
    def tile_log(self) -> Path:
        return self.tile_video.with_suffix('.log')
