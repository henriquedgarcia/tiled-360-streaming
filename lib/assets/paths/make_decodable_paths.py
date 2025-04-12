from pathlib import Path

from lib.assets.paths.makedashpaths import MakeDashPaths


class MakeDecodablePaths(MakeDashPaths):
    @property
    def decodable_folder(self) -> Path:
        return self.decodable_folder0 / self.folder_name_proj_tiling_tile_qlt

    @property
    def decodable_chunk(self) -> Path:
        return self.decodable_folder / f'chunk{self.chunk}.mp4'
