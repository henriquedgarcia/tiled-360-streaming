import json
from contextlib import contextmanager
from pathlib import Path

from lib.assets.autodict import AutoDict


class StatusCtx:
    status: dict
    cls_name: type

    def __init__(self, config, ctx):
        self.config = config
        self.ctx = ctx

    @contextmanager
    def status_context(self, cls_name=None):
        self.cls_name = cls_name
        self.load_status()

        try:
            yield
        finally:
            self.save_status()

    def load_status(self):
        try:
            self.status = json.loads(self.status_filename.read_text(),
                                     object_hook=lambda value: AutoDict(value))
        except FileNotFoundError:
            self.status = AutoDict()
            self.status_filename.write_text(json.dumps(self.status, indent=0, separators=(',', ':')))

    @property
    def status_filename(self):
        return Path(f'log/status_{self.config.project_folder.name}_{self.cls_name}.json')

    def update_status(self, key, value):
        """

        :param value:
        :param key:
        :return:
        """
        status = self.status
        for item in ['name', 'projection', 'quality', 'tiling', 'tile', 'chunk']:
            if status[item] is None: continue
            status = status[item]
        status[key] = value

    def get_status(self, key=None):
        status = self.status
        for item in self.ctx:
            status = status[item]
        if key is None:
            return status
        return status[key]

    def save_status(self):
        print('Saving Status.')
        self.status_filename.write_text(json.dumps(self.status, indent=None, separators=(',', ':')))