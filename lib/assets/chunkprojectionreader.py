from pathlib import Path

import numpy as np
from py360tools import ProjectionBase

from lib.utils.util import iter_video, make_tile_positions


class ChunkProjectionReader:
    tiles_reader: dict
    tile_positions: dict
    canvas: np.ndarray

    def __init__(self,
                 seen_tiles: dict[str, Path],
                 proj: ProjectionBase
                 ):
        """

        :param seen_tiles: Um dicionário do tipo {seen_tile: tile_file_path, ...} by chunk
        :param proj:
        """
        self.seen_tiles = seen_tiles
        self.proj = proj

        self.tile_positions = make_tile_positions(proj)
        self.canvas = np.zeros(self.proj.canvas.shape, dtype='uint8')
        self.tiles_reader = {str(seen_tile): iter_video(file_path, gray=True)
                             for seen_tile, file_path in self.seen_tiles.items()}

    def extract_viewport(self, yaw_pitch_roll) -> np.ndarray:
        self.mount_frame()

        viewport = self.proj.extract_viewport(self.canvas, yaw_pitch_roll)
        return viewport

    def mount_frame(self):
        self.canvas[:] = 0
        frame_idx = 0
        for tile in self.seen_tiles:
            frame_idx += 1
            x_ini, x_end, y_ini, y_end = self.tile_positions[tile]

            try:
                tile_frame = next(self.tiles_reader[tile])
            except Exception as e:
                raise e
            self.canvas[y_ini:y_end, x_ini:x_end] = tile_frame
