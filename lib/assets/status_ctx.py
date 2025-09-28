import json
from contextlib import contextmanager
from pathlib import Path

from lib.assets.autodict import AutoDict
from lib.utils.util import get_nested_value
from lib.utils.io_util import save_json


class StatusCtx:
    status: dict
    cls_name: type

    def __init__(self, ctx):
        self.config = ctx.config
        self.ctx = ctx

    @contextmanager
    def status_context(self, cls_name=None):
        self.cls_name = cls_name
        self.load_status()

        try:
            yield
        finally:
            print('Saving Status.')
            self.save_status()

    def load_status(self):
        try:
            self.status = json.loads(self.status_filename.read_text(),
                                     object_hook=lambda value: AutoDict(value))
        except FileNotFoundError:
            self.status = AutoDict()
            self.save_status()

    @property
    def status_filename(self):
        return Path(f'log/status_{self.cls_name}_{self.config.project_folder.name}.json')

    def update_status(self, key, value):
        """

        :param value:
        :param key:
        :return:
        """
        keys = [self.ctx.name, self.ctx.projection, self.ctx.quality, self.ctx.tiling, self.ctx.tile, self.ctx.chunk]
        get_nested_value(self.status, keys).update({key: value})

    def get_status(self, key=None):
        keys = [self.ctx.name, self.ctx.projection, self.ctx.quality, self.ctx.tiling, self.ctx.tile, self.ctx.chunk]
        return get_nested_value(self.status, keys)[key]

    def save_status(self):
        save_json(self.status, self.status_filename)
