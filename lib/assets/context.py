from collections import defaultdict
from functools import cached_property
from math import prod

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.utils.util import load_json, splitx


class Context:
    name: str = None
    projection: str = None
    quality: str = None
    tiling: str = None
    tile: str = None
    chunk: str = None
    frame: int = None
    user: str = None
    metric: str = None
    turn: str = None
    attempt: int = None
    iterations: int = 0
    projection_dict = AutoDict

    factors_list = ['name', 'projection', 'tiling', 'tile', 'user', 'quality', 'chunk',
                    'frame', 'metric', 'attempt']

    def __init__(self, config: Config):
        self.config = config

    def __iter__(self):
        """
        Iterates over the context variables. Remove attributes with '_' and None.
        :return:
        """
        itens = [getattr(self, key) for key in self.factors_list]
        return filter(lambda item: item is not None, itens)

    def __str__(self):
        txt = []
        for factor in self.factors_list:
            value = getattr(self, factor)
            if value is None:
                continue

            if factor in ['name', 'projection', 'tiling']:
                value = value
            elif factor == 'quality':
                value = f'{self.config.rate_control}{value}'
            elif factor == 'tile':
                value = f'tile{int(value):02d}'
            elif factor == 'chunk':
                value = f'chunk{int(value):02d}'
            elif factor == 'frame':
                value = f'frame{int(value):03d}'
            elif factor == 'user':
                value = f'user{int(value):02d}'
            elif factor == 'attempt':
                value = f'attempt{value}'
            else:
                continue
            txt.append(f'[{value}]')

        return ''.join(txt)

    @cached_property
    def name_list(self) -> list[str]:
        return [str(name) for name in self.config.videos_dict]

    @cached_property
    def projection_list(self):
        return list(self.config.config_dict['scale'])

    @cached_property
    def quality_list(self):
        return self.config.quality_list

    @cached_property
    def tiling_list(self):
        return [str(tiling) for tiling in self.config.tiling_list]

    @property
    def tile_list(self):
        return [str(tile) for tile in range(self.n_tiles)]

    @cached_property
    def chunk_list(self):
        return [str(chunk) for chunk in range(1, self.config.n_chunks + 1)]

    @cached_property
    def group_list(self):
        return list({self.config.videos_dict[video]['group']
                     for video in self.name_list})

    @cached_property
    def metric_list(self):
        return self.config.metric_list

    @cached_property
    def hmd_dataset(self):
        return load_json(self.config.dataset_file)

    @property
    def users_list(self):
        return list(map(str, range(60)))

    @property
    def users_list_by_name(self):
        users_str = self.hmd_dataset[self.name + '_nas'].keys()
        sorted_users_int = sorted(map(int, users_str))
        sorted_users_str = list(map(str, sorted_users_int))
        return sorted_users_str

    _name_list_by_user: dict = None

    @property
    def name_list_by_user(self):
        if self._name_list_by_user is None:
            self._name_list_by_user = defaultdict(list)
            for name in self.name_list:
                for user in self.hmd_dataset[name + '_nas']:
                    self._name_list_by_user[user].append(name)

        return self._name_list_by_user[self.user]

    @property
    def frame_list(self) -> list[str]:
        return [str(frame) for frame in range(self.config.n_frames)]

    @property
    def scale(self):
        return self.config.config_dict['scale'][self.projection]

    @property
    def fov(self):
        return self.config.config_dict['fov']

    @property
    def n_tiles(self):
        return prod(map(int, splitx(self.tiling)))

    @property
    def offset(self):
        return self.config.videos_dict[self.name]['offset']

    _group = None

    @property
    def group(self):
        if self._group is None:
            return self.config.videos_dict[self.name]['group']
        return self._group

    @group.setter
    def group(self, value):
        self._group = value

    @group.deleter
    def group(self):
        self._group = None

    @property
    def video_shape(self) -> tuple:
        video_w, video_h = splitx(self.scale)
        return video_h, video_w
