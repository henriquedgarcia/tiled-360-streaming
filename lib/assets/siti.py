from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from lib.utils.util import iter_video


class SiTi:
    filename: Path
    previous_frame: Optional[np.ndarray]
    siti: dict

    def __init__(self, filename: Path):
        self.siti = defaultdict(list)
        self.previous_frame = None

        for n, frame in enumerate(iter_video(filename)):
            si = self._calc_si(frame)
            self.siti['si'].append(si)

            ti = self._calc_ti(frame)
            self.siti['ti'].append(ti)

            print(f'\rSiTi - {filename.parts[-5:]}: frame={n}, si={si:.2f}, ti={ti:.3f}', end='')

        print('')

    def __getitem__(self, item) -> list:
        return self.siti[item]

    @staticmethod
    def _calc_si(frame: np.ndarray) -> (float, np.ndarray):
        """
        Calculate Spatial Information for a video frame. Calculate both vectors and so the magnitude.
        :param frame: A luma video frame in numpy ndarray format.
        :return: spatial information and sobel frame.
        """
        from PIL import Image
        sob_y = ndimage.sobel(frame, axis=0)
        sob_x = ndimage.sobel(frame, axis=1, mode="wrap")

        def f(sob):
            pos = sob - sob.min()
            nor = pos / pos.max()
            img = nor * 255
            Image.fromarray(img).show()

        sobel = np.hypot(sob_y, sob_x)
        si = sobel.std()
        return si

    def _calc_ti(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calculate Temporal Information for a video frame. If is a first frame,
        the information is zero.
        :param frame: A luma video frame in numpy ndarray format.
        :return: Temporal information and difference frame. If first frame the
        difference is zero array on same shape of frame.
        """
        try:
            difference = frame - self.previous_frame
        except TypeError:
            return 0.
        finally:
            self.previous_frame = frame
        return difference.std()
