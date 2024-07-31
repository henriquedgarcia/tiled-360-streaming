from math import prod
from pathlib import Path

from lib.assets.context import ctx
from lib.util import splitx
from .lazyproperty import LazyProperty


class Factors:
    config_dict: dict
    videos_dict: dict

    _quality_list: list
    _tiling_list: list

    @LazyProperty
    def project_folder(self):
        return Path(self.config_dict['project_folder'])

    @LazyProperty
    def duration(self):
        return self.config_dict['duration']

    @LazyProperty
    def fps(self):
        return self.config_dict['fps']

    @LazyProperty
    def gop(self):
        return self.config_dict['gop']

    @LazyProperty
    def codec(self):
        return self.config_dict['codec']

    @LazyProperty
    def codec_params(self):
        return self.config_dict['codec_params']

    @LazyProperty
    def rate_control(self):
        return self.config_dict['rate_control']

    @LazyProperty
    def original_quality(self):
        return self.config_dict['original_quality']

    @LazyProperty
    def decoding_num(self):
        return self.config_dict['decoding_num']

    @LazyProperty
    def error_metric(self):
        return self.config_dict['error_metric']

    @LazyProperty
    def distributions(self):
        return self.config_dict['distributions']

    @LazyProperty
    def fov(self):
        return self.config_dict['fov']

    @LazyProperty
    def bins(self):
        return self.config_dict['bins']

    @LazyProperty
    def n_frames(self) -> int:
        return int(self.duration) * int(self.fps)

    @LazyProperty
    def chunk_duration(self) -> int:
        return int(self.gop) // int(self.fps)

    @LazyProperty
    def _tiling_list(self):
        return self.config_dict['tiling_list']

    @LazyProperty
    def _quality_list(self):
        return self.config_dict['quality_list']

    @property
    def n_tiles(self):
        return prod(map(int, splitx(ctx.tiling)[::-1]))

    @property
    def scale(self):
        return self.config_dict['scale'][ctx.projection]

    @property
    def group(self):
        return self.videos_dict[ctx.video]['group']

    @property
    def video_shape(self) -> tuple:
        w, h = splitx(self.scale)
        return h, w

    @property
    def video_h(self) -> tuple:
        return self.video_shape[0]

    @property
    def video_w(self) -> tuple:
        return self.video_shape[1]

    @property
    def cmp_face_resolution(self) -> str:
        h, w, _ = self.cmp_face_shape
        return f'{w}x{h}'

    @property
    def cmp_face_shape(self) -> (int, int, int):
        h, w, c = self.video_shape
        return round(h / 2), round(w / 3), c  # </editor-fold>

