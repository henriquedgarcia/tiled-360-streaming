from math import prod

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.assets.lazyproperty import LazyProperty
from lib.utils.util import splitx, load_json


class Context:
    name: str = None
    projection: str = None
    quality: str = None
    tiling: str = None
    tile: str = None
    chunk: str = None
    frame: str = None
    user: str = None
    metric: str = None
    turn: str = None
    attempt: int = None
    projection_dict = AutoDict

    factors_list = ['name', 'projection', 'quality', 'tiling', 'tile', 'chunk',
                    'frame', 'user', 'metric', 'attempt']

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

            if factor == 'quality':
                value = f'{self.config.rate_control}' + value
            if factor == 'tile':
                value = 'tile' + value
            if factor == 'chunk':
                value = 'chunk' + value
            if factor == 'frame':
                value = 'frame' + value
            if factor == 'user':
                value = 'user' + str(value)
            if factor == 'turn':
                value = 'turn' + str(value)
            if factor == 'attempt':
                value = 'attempt' + str(value)

            txt.append(f'[{value}]')

        return ''.join(txt)

    @LazyProperty
    def name_list(self):
        return [str(name) for name in self.config.videos_dict]

    @LazyProperty
    def projection_list(self):
        return list(self.config.config_dict['scale'])

    @LazyProperty
    def quality_list(self):
        return [str(quality) for quality in self.config.quality_list]

    @LazyProperty
    def tiling_list(self):
        return [str(tiling) for tiling in self.config.tiling_list]

    @property
    def tile_list(self):
        return [str(tile) for tile in range(self.n_tiles)]

    @LazyProperty
    def chunk_list(self):
        return [str(chunk) for chunk in range(self.config.n_chunks)]

    @LazyProperty
    def group_list(self):
        return list({self.config.videos_dict[video]['group']
                     for video in self.name_list})

    @LazyProperty
    def metric_list(self):
        return self.config.metric_list

    hmd_dataset: dict = None

    @property
    def users_list(self):
        try:
            assert self.hmd_dataset is not None
        except AssertionError:
            self.hmd_dataset = load_json(self.config.dataset_file)

        users_str = self.hmd_dataset[self.name + '_nas'].keys()
        sorted_users_int = sorted(map(int, users_str))
        sorted_users_str = list(map(str, sorted_users_int))
        return sorted_users_str

    @property
    def frame_list(self) -> list[str]:
        return [str(frame) for frame in range(self.config.n_frames)]

    @property
    def scale(self):
        return self.config.config_dict['scale'][self.projection]

    @property
    def n_tiles(self):
        return prod(map(int, splitx(self.tiling)))

    @property
    def offset(self):
        return self.config.videos_dict[self.name]['offset']

    @property
    def group(self):
        return self.config.videos_dict[self.name]['group']

    @property
    def video_shape(self) -> tuple:
        video_w, video_h = splitx(self.scale)
        return video_h, video_w
