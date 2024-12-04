from pathlib import Path

import numpy as np
from skvideo.io import FFmpegReader

from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.utils.util import build_projection, idx2xy, iter_video


class MountFrame(CtxInterface):
    tiles_reader: dict
    canvas: np.ndarray

    def __init__(self, seen_tiles: dict[str, Path], ctx: Context):
        """

        :param seen_tiles: by chunk
        :param ctx:
        """
        self.seen_tiles = seen_tiles
        self.ctx = ctx
        self.proj = build_projection(proj_name=self.projection,
                                     tiling=self.tiling,
                                     proj_res=self.scale,
                                     vp_res='1320x1080',
                                     fov_res=self.fov)
        self.reset_readers()

    def reset_readers(self):
        self.tiles_reader = {}
        for seen_tile, file_path in self.seen_tiles.items():
            self.tiles_reader[seen_tile] = iter_video(file_path)

    def clear_frame(self):
        proj_h, proj_w = self.proj.canvas.shape
        self.canvas = np.zeros((proj_h, proj_w, 3), dtype='uint8')

    def get_frame(self):
        self.clear_frame()

        tile_h, tile_w = self.proj.tiling.tile_shape
        tile_N, tile_M = self.proj.tiling.shape

        for tile in self.seen_tiles:
            tile_m, tile_n = idx2xy(int(tile), (tile_N, tile_M))
            tile_y, tile_x = tile_n * tile_h, tile_m * tile_w
            y_ini = tile_y
            y_end = tile_y + tile_h
            x_ini = tile_x
            x_end = tile_x + tile_w

            tile_frame = next(self.tiles_reader[tile])
            self.canvas[y_ini:y_end, x_ini:x_end, :] = tile_frame
        return self.canvas
