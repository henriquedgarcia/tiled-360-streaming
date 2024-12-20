from pathlib import Path

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.basepaths import BasePaths
from lib.assets.paths.gettilespaths import GetTilesPaths
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths


class ViewportQualityPaths(CtxInterface):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.base_paths = BasePaths(ctx)
        self.decodable_paths = MakeDecodablePaths(ctx)
        self.get_tiles_paths = GetTilesPaths(self.ctx)
        self.chunk_quality_paths = ChunkQualityPaths(self.ctx)

    @property
    def viewport_quality_folder(self) -> Path:
        """
        Need None
        """
        return self.base_paths.viewport_quality_folder

    @property
    def user_viewport_quality_json(self) -> Path:
        """
        Need None
        """
        return self.viewport_quality_folder / f'{self.name}' / f'{self.tiling}' / f'{self.user}' / f'{self.ctx.config.rate_control}{self.quality}' / f'user_viewport_quality_chunk{self.chunk}.json'

    @property
    def get_tiles_result_json(self) -> Path:
        """
        Need None
        """
        return self.get_tiles_paths.get_tiles_result_json

    @property
    def decodable_chunk(self) -> Path:
        """
        Need None
        """
        return self.decodable_paths.decodable_chunk

    @property
    def reference_chunk(self):
        return self.chunk_quality_paths.reference_chunk
