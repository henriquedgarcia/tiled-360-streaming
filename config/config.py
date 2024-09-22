from pathlib import Path
from typing import Union

from lib.utils.util import load_json
from py360tools.utils import LazyProperty


class ConfigProps:
    config_dict: dict[str, Union[str, int, dict, list]] = {}
    videos_dict: dict

    @LazyProperty
    def project_folder(self):
        return Path(self.config_dict['project_folder'])

    @LazyProperty
    def sph_file(self):
        return Path(self.config_dict['sph_file'])

    @LazyProperty
    def dataset_file(self) -> Path:
        return Path(self.config_dict['dataset_file'])

    #####################
    @LazyProperty
    def duration(self):
        return int(self.config_dict['duration'])

    @LazyProperty
    def fps(self):
        return self.config_dict['fps']

    @LazyProperty
    def gop(self):
        return self.config_dict['gop']

    #####################
    @LazyProperty
    def rate_control(self):
        return self.config_dict['rate_control']

    @LazyProperty
    def decoding_num(self):
        return self.config_dict['decoding_num']

    #####################
    @LazyProperty
    def bins(self):
        return self.config_dict['bins']

    @LazyProperty
    def metric_list(self):
        return self.config_dict['metric_list']

    @LazyProperty
    def error_metric(self):
        return self.config_dict['error_metric']

    @LazyProperty
    def distributions(self):
        return self.config_dict['distributions']

    #####################
    @LazyProperty
    def fov(self):
        return self.config_dict['fov']

    @LazyProperty
    def quality_list(self) -> list[str]:
        return self.config_dict['quality_list']

    @LazyProperty
    def tiling_list(self) -> list[str]:
        return self.config_dict['tiling_list']


class Config(ConfigProps):
    def __init__(self, config_file, videos_file):
        """

        :param config_file: The config json file
        :type config_file: Path
        :param videos_file: The videos list json file
        :type videos_file: Path
        :return:
        """
        self.config_dict = load_json(config_file)
        self.videos_dict = load_json(videos_file)

    @LazyProperty
    def n_frames(self) -> int:
        return int(self.duration) * int(self.fps)

    @LazyProperty
    def chunk_duration(self) -> int:
        return int(self.gop) // int(self.fps)

    @LazyProperty
    def n_chunks(self) -> int:
        return self.n_frames // self.gop

    # @property
    # def cmp_face_shape(self) -> (int, int, int):
    #     h, w, c = self.video_shape
    #     return round(h / 2), round(w / 3), c
    #
    # @property
    # def cmp_face_resolution(self) -> str:
    #     h, w, _ = self.cmp_face_shape
    #     return f'{w}x{h}'


