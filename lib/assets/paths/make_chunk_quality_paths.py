from pathlib import Path

from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.utils.context_utils import context_quality


class MakeChunkQualityPaths(MakeDecodablePaths):
    @property
    def reference_chunk(self):
        with context_quality(self.ctx, '0', 'qp'):
            chunk_video_paths = self.decodable_chunk
        return chunk_video_paths

    @property
    def chunk_quality_result_by_name(self) -> Path:
        return self.project_path / f'quality/chunk_quality_{self.metric}_{self.name}_{self.projection}_{self.rate_control}.pickle'

    @property
    def chunk_quality_result(self) -> Path:
        """depend on name and fov"""
        return self.results_folder / f'chunk_quality_{self.metric}_{self.projection}_{self.rate_control}.pickle'

    @property
    def chunk_quality_folder(self) -> Path:
        folder = self.quality_folder / self.folder_name_proj_tiling_tile
        return folder

    @property
    def chunk_quality_pickle(self) -> Path:
        filename = f'chunk{int(self.chunk):03d}_{self.config.rate_control}{self.ctx.quality}.pickle'
        return self.chunk_quality_folder / filename

    @property
    def chunk_quality_json(self) -> Path:
        filename = f'chunk{int(self.chunk):03d}_{self.config.rate_control}{self.ctx.quality}.json'
        return self.chunk_quality_folder / filename

    @property
    def quality_result_img(self) -> Path:
        folder = self.quality_folder / '_metric plots' / f'{self.name}'
        return folder / f'{self.tiling}_crf{self.quality}.png'
