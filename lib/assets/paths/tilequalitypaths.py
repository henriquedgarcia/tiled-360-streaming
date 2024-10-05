from pathlib import Path

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.utils.context_utils import context_quality


class ChunkQualityPaths(CtxInterface):
    def __init__(self, context: Context):
        self.ctx = context
        self.config = context.config
        self.base_paths = BasePaths(context)
        self.segmenter_paths = SegmenterPaths(context)

    @property
    def reference_chunk(self):
        with context_quality(self.ctx, '0', 'qp'):
            chunk_video_paths = self.segmenter_paths.chunk_video
        return chunk_video_paths

    @property
    def chunk_quality_result_json(self) -> Path:
        return self.base_paths.results_folder / f'chunk_quality_{self.name}.json'

    @property
    def chunk_quality_folder(self) -> Path:
        folder = self.base_paths.quality_folder / self.base_paths.basename2
        return folder

    @property
    def chunk_quality_json(self) -> Path:
        filename = f'tile{self.tile}_{int(self.chunk):03d}.json'
        return self.chunk_quality_folder / filename

    @property
    def quality_result_img(self) -> Path:
        folder = self.base_paths.quality_folder / '_metric plots' / f'{self.name}'
        return folder / f'{self.tiling}_crf{self.quality}.png'
