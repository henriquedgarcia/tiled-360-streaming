from .config import config


class Context:
    def __init__(self):
        self.projection = None
        self.video = None
        self.tiling = None
        self.quality = None
        self.tile = None
        self.chunk = None
        self.metric = None
        self.user = None
        self.turn = None

    @property
    def quality_str(self) -> str:
        return f'{config.rate_control}{self.quality}'

    @property
    def chunk_str(self) -> str:
        return f'chunk{self.chunk}'

    @property
    def tile_str(self) -> str:
        return f'tile{self.tile}'


ctx = Context()
