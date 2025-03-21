from pathlib import Path

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.utils.context_utils import context_quality


class ChunkQualityPaths(CtxInterface):
    def __init__(self, context: Context):
        self.ctx = context
        self.base_paths = BasePaths(context)
        self.make_decodable_paths = MakeDecodablePaths(context)

    @property
    def reference_chunk(self):
        with context_quality(self.ctx, '0', 'qp'):
            chunk_video_paths = self.decodable_chunk
        return chunk_video_paths

    @property
    def decodable_chunk(self):
        return self.make_decodable_paths.decodable_chunk

    @property
    def chunk_quality_result_json(self) -> Path:
        return self.base_paths.results_folder / f'quality/chunk_quality_{self.name}.json'

    @property
    def chunk_quality_result_pickle(self) -> Path:
        return self.base_paths.results_folder / f'quality/chunk_quality_serie_{self.metric}_{self.name}.pickle'

    @property
    def chunk_quality_folder(self) -> Path:
        folder = self.base_paths.quality_folder / self.base_paths.basename_lv4
        return folder

    @property
    def chunk_quality_json(self) -> Path:
        filename = f'chunk{int(self.chunk):03d}_{self.config.rate_control}{self.ctx.quality}.json'
        return self.chunk_quality_folder / filename

    @property
    def quality_result_img(self) -> Path:
        folder = self.base_paths.quality_folder / '_metric plots' / f'{self.name}'
        return folder / f'{self.tiling}_crf{self.quality}.png'
