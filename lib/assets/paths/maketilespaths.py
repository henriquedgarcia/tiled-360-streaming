from pathlib import Path

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths


class MakeTilesPaths(BasePaths):
    @property
    def lossless_video(self) -> Path:
        return self.lossless_folder / self.projection / f'{self.name}.mp4'

    @property
    def lossless_log(self) -> Path:
        return self.lossless_video.with_suffix('.log')

    @property
    def tile_folder(self) -> Path:
        folder = self.tiles_folder / self.folder_name_proj_tiling_tile
        return folder

    @property
    def tile_video(self) -> Path:
        return self.tile_folder / f'tile{self.tile}_{self.config.rate_control}{self.ctx.quality}.mp4'

    @property
    def tile_log(self) -> Path:
        return self.tile_video.with_suffix('.log')
