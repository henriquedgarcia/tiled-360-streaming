from functools import cached_property

import numpy as np
from py360tools.utils.util import splitx

from .projectionframe import ProjectionFrame
from .tile import Tile


class Tiling:
    tiling: str
    proj_res: str
    shape: np.ndarray
    ntiles: int
    tile_shape: np.ndarray

    def __init__(self, tiling, projection_frame):
        """

        :param tiling: "12x8"
        :type tiling: str
        :param projection_frame: "12x8"
        :type projection_frame: ProjectionFrame
        """
        self.tiling = tiling
        self.proj_res = projection_frame.proj_res
        self.shape = np.array(splitx(tiling)[::-1])
        self.ntiles = self.shape[0] * self.shape[1]
        self.tile_shape = (projection_frame.shape / self.shape).astype(int)

    @cached_property
    def tile_list(self):
        return [Tile(tile_id, self) for tile_id in range(self.ntiles)]

    def __str__(self):
        return self.tiling

    def __repr__(self):
        return f'{self.__class__.__name__}({self}@{self.proj_res})'

    def __eq__(self, other: "Tiling"):
        return repr(self) == repr(other)
