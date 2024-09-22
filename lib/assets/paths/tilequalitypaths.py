from pathlib import Path

from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.utils.context_utils import context_quality


class TileChunkQualityPaths:
    def __init__(self, config, ctx):
        self.config = config
        self.ctx = ctx
        self.base_paths = BasePaths(self.config, self.ctx)
        self.segmenter_paths = SegmenterPaths(self.config, self.ctx)

    @property
    def reference_chunk(self):
        with context_quality(self.ctx, self.config, '0', 'crf'):
            chunk_video_paths = self.segmenter_paths.chunk_video
        return chunk_video_paths

    @property
    def video_quality_json(self) -> Path:
        return self.base_paths.results_folder / f'tile_chunk_quality_{self.ctx.name}.json'

    @property
    def tile_chunk_quality_folder(self) -> Path:
        folder = self.base_paths.quality_folder / self.base_paths.basename2
        return folder

    @property
    def tile_chunk_quality_csv(self) -> Path:
        filename = f'tile{self.ctx.tile}_{int(self.ctx.chunk):03d}.csv'
        return self.tile_chunk_quality_folder / filename

    @property
    def quality_result_img(self) -> Path:
        folder = self.base_paths.quality_folder / '_metric plots' / f'{self.ctx.name}'
        return folder / f'{self.ctx.tiling}_crf{self.ctx.quality}.png'
