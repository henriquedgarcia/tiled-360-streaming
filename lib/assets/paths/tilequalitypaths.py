from pathlib import Path

from lib.assets.context import ctx
from lib.assets.paths import segmenter_paths, base_paths
from lib.utils.context_utils import context_quality


class TileQualityPaths:
    @property
    def reference_chunk(self):
        with context_quality('0', 'crf'):
            chunk_video_paths = segmenter_paths.chunk_video
        return chunk_video_paths

    @property
    def quality_result_json(self) -> Path:
        return base_paths.results_folder / f'quality_{ctx.name}.json'


tilequality_paths = TileQualityPaths()
