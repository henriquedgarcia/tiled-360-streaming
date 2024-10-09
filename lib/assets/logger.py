from time import time
import datetime
from collections import defaultdict
from contextlib import contextmanager

import pandas as pd


class Logger:
    _log: defaultdict
    cls_name: str

    def __init__(self, ctx):
        self.config = ctx.config
        self.ctx = ctx

    @contextmanager
    def logger_context(self, cls_name):
        self.cls_name = cls_name
        self._log = defaultdict(list)
        start = time()

        try:
            yield
        finally:
            self.save_log()
            print(f"\n\tTotal time={time() - start}.")

    def register_log(self, error_code: str, filepath):
        self._log['name'].append(f'{self.ctx.name}')
        self._log['tiling'].append(f'{self.ctx.tiling}')
        self._log['quality'].append(f'{self.ctx.quality}')
        self._log['tile'].append(f'{self.ctx.tile}')
        self._log['chunk'].append(f'{self.ctx.chunk}')
        self._log['error'].append(error_code)
        self._log['parent'].append(f'{filepath.parent}')
        self._log['path'].append(f'{filepath.absolute()}')

    def save_log(self):
        print('Saving Log.')
        now = f'{datetime.datetime.now()}'.replace(':', '-')
        filename = f'log/log_{self.cls_name}_{now}.csv'
        df_log_text = pd.DataFrame(self._log)
        df_log_text.to_csv(filename, encoding='utf-8')
