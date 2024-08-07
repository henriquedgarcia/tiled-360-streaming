from .config import config
from .lazyproperty import LazyProperty


def get_str_prefix(item):
    if item in ['tile', 'chunk']:
        prefix = item
    elif item == 'quality':
        prefix = config.rate_control
    else:
        prefix = ''
    return prefix


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

    factors_list = ['name', 'projection', 'quality', 'tiling', 'tile', 'chunk',
                    'user', 'metric', 'turn']

    def __iter__(self):
        """
        Iterates over the context variables. Remove attributes with '_' and None.
        :return:
        """
        itens = [getattr(self, key) for key in self.factors_list]
        return filter(lambda item: item is not None, itens)

    def __str__(self):
        txt = []
        for key in self.factors_list:
            item = getattr(self, key)
            if item is None:
                continue

            txt.append(f'[{get_str_prefix(item)}{item}]')

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
        return [str(tile) for tile in range(config.n_tiles)]

    @LazyProperty
    def chunk_list(self):
        return [str(chunk) for chunk in range(1, config.duration + 1)]

    @LazyProperty
    def group_list(self):
        return list({config.videos_dict[video]['group']
                     for video in self.name_list})

    @LazyProperty
    def metric_list(self):
        return config.metric_list

    @property
    def user_list(self):
        return [str(user) for user in []]

    @property
    def frame_list(self) -> list[str]:
        return [str(frame) for frame in range(config.n_frames)]


ctx = Context()
