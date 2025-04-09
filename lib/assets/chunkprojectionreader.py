from math import prod
from pathlib import Path

import numpy as np
from py360tools import CMP, ERP, ProjectionBase

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.utils.util import build_projection, idx2xy, iter_video, splitx


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
        """
        self.seen_tiles = seen_tiles
        self.tile_list = list(map(str, range(prod(splitx(proj.tiling.tiling)))))
        self.proj = proj
        self.reset_readers()
        self.clear_frame()
        self.make_tile_positions()

    def reset_readers(self):
        self.tiles_reader = {}
        for seen_tile, file_path in self.seen_tiles.items():
            self.tiles_reader[seen_tile] = iter_video(file_path, gray=True)

    def clear_frame(self):
        proj_h, proj_w = self.proj.canvas.shape
        self.canvas = np.zeros((proj_h, proj_w), dtype='uint8')

    def get_frame(self) -> np.ndarray:
        self.canvas[:] = 0

        for tile in self.seen_tiles:
            x_ini, x_end, y_ini, y_end = self.tile_positions[tile]
            tile_frame = next(self.tiles_reader[tile])
            self.canvas[y_ini:y_end, x_ini:x_end] = tile_frame
        return self.canvas

    def extract_viewport(self, yaw_pitch_roll) -> np.ndarray:
        canvas = self.get_frame()
        viewport = self.proj.extract_viewport(canvas, yaw_pitch_roll)
        return viewport

    def make_tile_positions(self):
        self.tile_positions = {}
        tile_h, tile_w = self.proj.tiling.tile_shape
        tile_N, tile_M = self.proj.tiling.shape

        for tile in self.tile_list:
            tile_m, tile_n = idx2xy(int(tile), (tile_N, tile_M))
            tile_y, tile_x = tile_n * tile_h, tile_m * tile_w
            x_ini = tile_x
            x_end = tile_x + tile_w
            y_ini = tile_y
            y_end = tile_y + tile_h
            self.tile_positions[tile] = x_ini, x_end, y_ini, y_end
