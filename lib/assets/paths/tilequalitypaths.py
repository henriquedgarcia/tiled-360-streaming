from pathlib import Path

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.utils.context_utils import context_quality


class ChunkQualityPaths(BasePaths):
    def __init__(self, context: Context):
        self.ctx = context
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
        return self.results_folder / f'quality/chunk_quality_{self.name}.json'

    @property
    def chunk_quality_result_pickle(self) -> Path:
        return self.results_folder / f'quality/{self.metric}_{self.name}.pickle'

    @property
    def chunk_quality_folder(self) -> Path:
        folder = self.quality_folder / self.folder_name_proj_tiling_tile
        return folder

    @property
    def chunk_quality_json(self) -> Path:
        filename = f'chunk{int(self.chunk):03d}_{self.config.rate_control}{self.ctx.quality}.json'
        return self.chunk_quality_folder / filename

    @property
    def quality_result_img(self) -> Path:
        folder = self.quality_folder / '_metric plots' / f'{self.name}'
        return folder / f'{self.tiling}_crf{self.quality}.png'
