from typing import Union

import numpy as np
from PIL import Image

from .projectionbase import ProjBase, compose


class ERP(ProjBase):
    def nm2xyz(self, nm: np.ndarray,
               proj_shape: Union[np.ndarray, tuple]) -> np.ndarray:
        if proj_shape is None:
            proj_shape = nm.shape[1:]

        vu = self.erp2vu(nm, proj_shape=proj_shape)
        ea = self.vu2ea(vu)
        xyz = self.ea2xyz(ea)  # common
        return xyz

    @staticmethod
    def erp2vu(nm: np.ndarray, proj_shape=None) -> np.ndarray:
        if proj_shape is None:
            proj_shape = nm.shape[1:]
        vu = (nm + [[[0.5]], [[0.5]]]) / [[[proj_shape[0]]], [[proj_shape[1]]]]
        return vu

    @staticmethod
    def vu2ea(vu: np.ndarray) -> np.ndarray:
        ea = (vu * [[[-np.pi]], [[2 * np.pi]]]) + [[[np.pi / 2]], [[-np.pi]]]
        return ea

    ###########################################################

    def xyz2nm(self, xyz: np.ndarray,
               proj_shape: np.ndarray = None):
        """
        ERP specific.

        :param xyz: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param proj_shape: the shape of projection that cover all sphere. tuple as (N, M)
        :return:
        """
        if proj_shape is None:
            proj_shape = xyz.shape[:2]

        ea = self.xyz2ea(xyz)
        ea = self.normalize_ea(ea)
        vu = self.ea2vu(ea)
        nm = self.vu2nm(vu, proj_shape)

        return nm

    @staticmethod
    def ea2vu(ea):
        """

        :param ea: shape==(2,...)
        :return:
        """

        vu = np.zeros(ea)
        vu[0] = -ea[0] / np.pi + 0.5
        vu[1] = ea[1] / (2 * np.pi) + 0.5
        return vu

    @staticmethod
    def vu2nm(vu, proj_shape=None):
        if proj_shape is None:
            proj_shape = vu.shape[1:]

        nm = vu * [[[proj_shape[0]]], [[proj_shape[1]]]]
        nm = np.ceil(nm)
        return nm.astype(int)

    #############################


def test_erp():
    # erp '144x72', '288x144','432x216','576x288'
    yaw_pitch_roll = np.deg2rad((70, 0, 0))
    height, width = 288, 576

    ########################################
    # Open Image
    frame_img: Union[Image, list] = Image.open('images/erp1.jpg')
    frame_img = frame_img.resize((width, height))

    erp = ERP(tiling='6x4', proj_res=f'{width}x{height}', fov='100x90')
    erp.yaw_pitch_roll = yaw_pitch_roll
    compose(erp, frame_img)


if __name__ == '__main__':
    test_erp()
