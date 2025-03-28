from pathlib import Path

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.paths.seen_tiles_paths import SeenTilesPaths
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths


class ViewportQualityPaths(BasePaths):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.decodable_paths = MakeDecodablePaths(ctx)
        self.get_tiles_paths = SeenTilesPaths(self.ctx)
        self.chunk_quality_paths = ChunkQualityPaths(self.ctx)

    @property
    def user_viewport_quality_json(self) -> Path:
        """
        Need name tiling user quality chunk
        """
        return (self.viewport_quality_folder /
                f'{self.name}' / f'{self.tiling}' / f'{self.user}' / f'{self.quality}' /
                f'user_viewport_quality_chunk{self.chunk}.json')

    @property
    def user_viewport_quality_result_json(self) -> Path:
        """
        Need name tiling user quality chunk
        """

        return self.results_folder / f'user_viewport_quality/user_viewport_quality_{self.name}.json'

    @property
    def get_tiles_result_json(self) -> Path:
        """
        Need name fov
        """
        return self.get_tiles_paths.seen_tiles_result_json

    @property
    def decodable_chunk(self) -> Path:
        """
        Need all
        """
        return self.decodable_paths.decodable_chunk

    @property
    def reference_chunk(self):
        """
        all minus quality
        :return:
        """
        return self.chunk_quality_paths.reference_chunk
