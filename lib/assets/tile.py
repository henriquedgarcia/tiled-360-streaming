import numpy as np

from py360tools.utils.util_transform import get_tile_borders


class Tile:
    shape: np.ndarray = None
    borders_nm: np.ndarray = None
    position_nm: np.ndarray = None

    def __init__(self, tile_id, tiling):
        """
        # Property
        tiling_position: position of tile in the tiling array
        position: position of tile in the projection image

        :param tile_id: A number on int or str
        :type tile_id: int | str
        :param tiling: Tiling("12x6", projection)
        :type tiling: Tiling
        """
        self.tile_id = int(tile_id)
        self.tiling = str(tiling)
        self.shape = tiling.tile_shape
        self.borders_nm = get_tile_borders(tile_id, tiling.shape, self.shape)
        self.position_nm = self.borders_nm[::, 0]

    def __str__(self):
        return f'tile{self.tile_id}'

    def __repr__(self):
        return f'tile{self.tile_id}@{self.tiling}'

    def __int__(self):
        return self.tile_id

    def __eq__(self, other):
        return (self.tile_id == other.tile_id and
                self.tiling == other.tiling and
                self.shape == other.shape)
