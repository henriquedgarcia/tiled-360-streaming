from functools import cached_property
from pathlib import Path
from typing import Union

from lib.utils.util import load_json


class ConfigProps:
    config_dict: dict[str, Union[str, int, dict, list]] = {}
    videos_dict: dict

    @cached_property
    def project_folder(self):
        return Path(self.config_dict['project_folder'])

    @cached_property
    def sph_file(self):
        return Path(self.config_dict['sph_file'])

    @cached_property
    def dataset_file(self) -> Path:
        return Path(self.config_dict['dataset_file'])

    #####################
    @cached_property
    def duration(self):
        return int(self.config_dict['duration'])

    @cached_property
    def fps(self):
        return self.config_dict['fps']

    @cached_property
    def gop(self):
        return self.config_dict['gop']

    #####################
    @cached_property
    def rate_control(self):
        return self.config_dict['rate_control']

    @cached_property
    def decoding_num(self):
        return self.config_dict['decoding_num']

    #####################
    @cached_property
    def bins(self):
        return self.config_dict['bins']

    @cached_property
    def metric_list(self):
        return self.config_dict['metric_list']

    @cached_property
    def error_metric(self):
        return self.config_dict['error_metric']

    @cached_property
    def distributions(self):
        return self.config_dict['distributions']

    #####################
    @cached_property
    def fov(self):
        return self.config_dict['fov']

    @cached_property
    def vp_res(self):
        return self.config_dict['vp_res']

    @cached_property
    def quality_list(self) -> list[str]:
        return self.config_dict['quality_list']

    @cached_property
    def tiling_list(self) -> list[str]:
        return self.config_dict['tiling_list']


class Config(ConfigProps):
    remove = False

    def __init__(self, config_file, videos_file):
        """

        :param config_file: The config JSON file
        :type config_file: Path
        :param videos_file: The videos list JSON file
        :type videos_file: Path
        :return:
        """
        self.config_dict = load_json(config_file)
        self.videos_dict = load_json(videos_file)

    @cached_property
    def n_frames(self) -> int:
        return int(self.duration) * int(self.fps)

    @cached_property
    def chunk_duration(self) -> int:
        return int(self.gop) // int(self.fps)

    @cached_property
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
