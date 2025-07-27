from pathlib import Path

import numpy as np
from py360tools import Viewport

from lib.utils.util import iter_video, make_tile_positions, idx2xy


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
        self.reset_readers()

    def clean_canvas(self):
        self.canvas[:] = 0

    def reset_readers(self):
        self.tiles_reader = {seen_tile: iter_video(file_path, gray=True)
                             for seen_tile, file_path in self.seen_tiles.items()}

    def mount_frame(self):
        self.clean_canvas()
        for tile in self.seen_tiles:
            tile_frame = next(self.tiles_reader[tile])

            x_ini, x_end, y_ini, y_end = self.tile_positions[int(tile)]
            self.canvas[y_ini:y_end, x_ini:x_end] = tile_frame
        return self.canvas

    def extract_viewport(self, yaw_pitch_roll) -> np.ndarray:
        self.mount_frame()
        viewport = self.vp.extract_viewport(self.canvas, yaw_pitch_roll)
        return viewport

    @staticmethod
    def make_tile_positions(viewport: Viewport) -> dict[int, tuple[int, int, int, int]]:
        """
        Um dicionário do tipo {tile: (x_ini, x_end, y_ini, y_end)}
        onde tiles é XXXX (verificar)
        e as coordenadas são inteiros.

        Mostra a posição inicial e final do tile na projeção.
        :param viewport:
        :return:
        """
        proj = viewport.projection
        tile_positions = {}
        tile_h, tile_w = proj.tile_shape
        tile_N, tile_M = proj.tiling_shape

        tile_list = list(proj.tile_list)

        for tile in tile_list:
            tile_m, tile_n = idx2xy(tile, (tile_N, tile_M))
            tile_y, tile_x = tile_n * tile_h, tile_m * tile_w
            x_ini = tile_x
            x_end = tile_x + tile_w
            y_ini = tile_y
            y_end = tile_y + tile_h
            tile_positions[tile] = x_ini, x_end, y_ini, y_end
        return tile_positions
