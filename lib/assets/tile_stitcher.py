from pathlib import Path

import numpy as np
from PIL import Image
from py360tools import Viewport

from lib.utils.util import iter_video, make_tiles_position, idx2xy


class TileStitcher:
    tiles_reader: dict

    def __init__(self,
                 tile_list: dict[str, Path],
                 viewport: Viewport,
                 full=False):
        """

        :param tile_list: Um dicionário do tipo {seen_tile: tile_file_path, ...} by chunk
        :param viewport:
        """
        self.tile_list = tile_list
        self.vp = viewport
        self.proj = viewport.projection
        self.tile_positions = make_tiles_position(self.proj)
        self.canvas = np.zeros(self.proj.shape, dtype='uint8')
        self.reset_readers()
        if full:
            self.load_all()

    full: np.ndarray
    
    def load_all(self):
        list_frames = []
        while True:
            try:
                canvas = self.next_proj_frame()
                list_frames.append(canvas.copy())
            except StopIteration:
                break
        self.full = np.stack(list_frames)
        self.reset_readers()
        return self.full

    def reset_readers(self):
        self.tiles_reader = {seen_tile: iter_video(file_path, gray=True)
                             for seen_tile, file_path in self.tile_list.items()}

    def clean_canvas(self):
        self.canvas[:] = 0

    def next_proj_frame(self):
        self.clean_canvas()
        for tile in self.tile_list:
            tile_frame = next(self.tiles_reader[tile])
            show(tile_frame)
            x_ini, x_end, y_ini, y_end = self.tile_positions[str(tile)]
            self.canvas[y_ini:y_end, x_ini:x_end] = tile_frame
        return self.canvas

    def extract_viewport(self, yaw_pitch_roll) -> np.ndarray:
        canvas = self.next_proj_frame()
        viewport = self.vp.extract_viewport(canvas, yaw_pitch_roll)
        return viewport

def show(frame1, frame2=None):
    if frame2 is None:
        Image.fromarray(frame1).show()
    else:
        Image.fromarray(np.abs(frame1 - frame2)).show()
