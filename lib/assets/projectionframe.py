import numpy as np

from py360tools.utils.util import splitx


class ProjectionFrame:
    def __init__(self, proj_res):
        """

        :param proj_res: A string representing the projection resolution. e.g. '600x3000'
        :type proj_res: str
        """

        # build_projection
        self.proj_res = proj_res
        self.shape = np.array(splitx(self.proj_res)[::-1], dtype=int)
