from time import time
import datetime
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from lib.assets.context import ctx
from config.config import config
from .autodict import AutoDict


class Logger:
    _log: defaultdict
    cls_name: str
    status: AutoDict

    def load_status(self):
        try:
            self.status = json.loads(self.status_filename.read_text(),
                                     object_hook=lambda value: AutoDict(value))
        except FileNotFoundError:
            self.status = AutoDict()

    @property
    def status_filename(self):
        return Path(f'log/status_{config.project_folder.name}_{self.cls_name}.json')

    @contextmanager
    def logger_context(self, cls_name):
        self.cls_name = cls_name
        self._log = defaultdict(list)
        self.load_status()
        start = time()

        try:
            yield
        finally:
            self.save_log()
            self.save_status()
            print(f"\n\tTotal time={time() - start}.")

    def register_log(self, error_code: str, filepath):
        self._log['name'].append(f'{ctx.name}')
        self._log['tiling'].append(f'{ctx.tiling}')
        self._log['quality'].append(f'{ctx.quality}')
        self._log['tile'].append(f'{ctx.tile}')
        self._log['chunk'].append(f'{ctx.chunk}')
        self._log['error'].append(error_code)
        self._log['parent'].append(f'{filepath.parent}')
        self._log['path'].append(f'{filepath.absolute()}')

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
        for item in ctx:
            status = status[item]
        if key is None:
            return status
        return status[key]

    def save_log(self):
        now = f'{datetime.datetime.now()}'.replace(':', '-')
        filename = f'log/log_{self.cls_name}_{now}.csv'
        df_log_text = pd.DataFrame(self._log)
        df_log_text.to_csv(filename, encoding='utf-8')

    def save_status(self):
        self.status_filename.write_text(json.dumps(self.status, indent=0, separators=(',', ':')))


logger = Logger()
