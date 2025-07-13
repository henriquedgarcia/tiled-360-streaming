from pathlib import Path

import numpy as np
from py360tools import Viewport

from lib.utils.util import iter_video, make_tile_positions


class ChunkProjectionReader:
    tiles_reader: dict
    tile_positions: dict
    canvas: np.ndarray

    def __init__(self,
                 seen_tiles: dict[str, Path],
                 viewport: Viewport
                 ):
        """

        :param seen_tiles: Um dicionário do tipo {seen_tile: tile_file_path, ...} by chunk
        :param viewport:
        """
        self.seen_tiles = seen_tiles
        self.vp = viewport
        self.proj = viewport.projection
        self.tile_positions = make_tile_positions(self.proj)
        self.canvas = np.zeros(self.proj.shape, dtype='uint8')
        self.tiles_reader = {seen_tile: iter_video(file_path, gray=True)
                             for seen_tile, file_path in self.seen_tiles.items()}

    def extract_viewport(self, yaw_pitch_roll) -> np.ndarray:
        self.mount_frame()
        viewport = self.vp.extract_viewport(self.canvas, yaw_pitch_roll)
        return viewport

    def mount_frame(self):
        self.canvas[:] = 0
        frame_idx = 0
        for tile in self.seen_tiles:
            frame_idx += 1
            x_ini, x_end, y_ini, y_end = self.tile_positions[int(tile)]

            try:
                tile_frame = next(self.tiles_reader[tile])
            except Exception as e:
                raise e
            self.canvas[y_ini:y_end, x_ini:x_end] = tile_frame
