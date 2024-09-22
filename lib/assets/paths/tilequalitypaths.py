from pathlib import Path

from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.utils.context_utils import context_quality


class TileQualityPaths:
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
    def quality_result_json(self) -> Path:
        return self.base_paths.results_folder / f'quality_{self.ctx.name}.json'


