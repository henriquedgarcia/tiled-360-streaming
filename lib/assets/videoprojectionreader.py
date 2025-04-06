from math import prod
from pathlib import Path

import numpy as np
from py360tools import CMP, ERP

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.utils.util import build_projection, idx2xy, iter_video, splitx


class VideoProjectionReader:
    tiles_reader: dict
    tile_positions: dict
    canvas: np.ndarray

    def __init__(self,
                 seen_tiles: dict[str, Path],
                 projection: str,
                 tiling: str,
                 proj_res: str,
                 fov_res: str,
                 vp_res: str,
                 ):
        """

        :param seen_tiles: by chunk
        :param ctx:
        """
        self.seen_tiles = seen_tiles
        self.tile_list = list(map(str, range(prod(splitx(tiling)))))
        if projection == 'erp':
            self.proj = ERP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res)
        elif projection == 'cmp':
            self.proj = CMP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res)

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
