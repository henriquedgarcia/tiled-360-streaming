from pathlib import Path

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths


class SegmenterPaths:
    def __init__(self, context: Context):
        self.config = context.config
        self.ctx = context
        self.base_paths = BasePaths(context)

    @property
    def lossless_video(self) -> Path:
        return self.base_paths.lossless_folder / self.ctx.projection / f'{self.ctx.name}.mp4'

    @property
    def lossless_log(self) -> Path:
        return self.lossless_video.with_suffix('.log')

    @property
    def tile_folder(self) -> Path:
        folder = self.base_paths.tiles_folder / self.base_paths.basename1
        return folder

    @property
    def tile_video(self) -> Path:
        return self.base_paths.tiles_folder / self.base_paths.basename1 / f'tile{self.ctx.tile}.mp4'

    @property
    def tile_log(self) -> Path:
        return self.tile_video.with_suffix('.log')

    @property
    def segmenter_log(self) -> Path:
        return self.base_paths.segmenter_folder / self.base_paths.basename1 / f'tile{self.ctx.tile}_segmenter.log'

    @property
    def chunks_folder(self) -> Path:
        return self.base_paths.segmenter_folder / self.base_paths.basename2

    @property
    def chunk_video(self) -> Path:
        chunk = int(self.ctx.chunk)
        return self.chunks_folder / f'tile{self.ctx.tile}_{chunk:03d}.hevc'

    @property
    def bitrate_result_json(self) -> Path:
        return self.base_paths.results_folder / f'rate_{self.ctx.name}.json'
