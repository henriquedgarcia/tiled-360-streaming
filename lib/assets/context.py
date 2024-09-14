from math import prod

from config.config import config
from lib.utils.util import splitx
from .lazyproperty import LazyProperty


class Context:
    name: str = None
    projection: str = None
    quality: str = None
    tiling: str = None
    tile: str = None
    chunk: str = None
    user: str = None
    metric: str = None
    turn: str = None
    attempt: int = None
    projection_obj = None

    factors_list = ['name', 'projection', 'quality', 'tiling', 'tile', 'chunk',
                    'user', 'metric', 'attempt']

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
                value = f'{config.rate_control}' + value
            if factor == 'tile':
                value = 'tile' + value
            if factor == 'chunk':
                value = 'chunk' + value
            if factor == 'turn':
                value = 'turn' + str(value)
            if factor == 'attempt':
                value = 'attempt' + str(value)

            txt.append(f'[{value}]')

        return ''.join(txt)

    @LazyProperty
    def name_list(self):
        return [str(name) for name in config.videos_dict]

    @LazyProperty
    def projection_list(self):
        return list(config.config_dict['scale'])

    @LazyProperty
    def quality_list(self):
        return [str(quality) for quality in config.quality_list]

    @LazyProperty
    def tiling_list(self):
        return [str(tiling) for tiling in config.tiling_list]

    @property
    def tile_list(self):
        return [str(tile) for tile in range(ctx.n_tiles)]

    @LazyProperty
    def chunk_list(self):
        return [str(chunk) for chunk in range(config.n_chunks)]

    @LazyProperty
    def group_list(self):
        return list({config.videos_dict[video]['group']
                     for video in self.name_list})

    @LazyProperty
    def metric_list(self):
        return config.metric_list

    hmd_dataset: dict = None

    @property
    def user_list(self):
        return [str(user) for user in []]

    @property
    def frame_list(self) -> list[str]:
        return [str(frame) for frame in range(config.n_frames)]

    @property
    def scale(self):
        return config.config_dict['scale'][self.projection]

    @property
    def n_tiles(self):
        return prod(map(int, splitx(self.tiling)))

    @property
    def offset(self):
        return config.videos_dict[self.name]['offset']

    @property
    def group(self):
        return config.videos_dict[self.name]['group']

    @property
    def video_shape(self) -> tuple:
        video_w, video_h = splitx(ctx.scale)
        return video_h, video_w


ctx = Context()
