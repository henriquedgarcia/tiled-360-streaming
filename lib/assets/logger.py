import datetime
from collections import defaultdict
from contextlib import contextmanager

import pandas as pd

from lib.assets.context import ctx


class Logger():
    log_text: defaultdict

    @contextmanager
    def logger(self):
        self.log_text = defaultdict(list)

        try:
            yield
        finally:
            self.save_log()

    def log(self, error_code: str, filepath):
        self.log_text['name'].append(f'{ctx.video}')
        self.log_text['tiling'].append(f'{ctx.tiling}')
        self.log_text['quality'].append(f'{ctx.quality}')
        self.log_text['tile'].append(f'{ctx.tile}')
        self.log_text['chunk'].append(f'{ctx.chunk}')
        self.log_text['error'].append(error_code)
        self.log_text['parent'].append(f'{filepath.parent}')
        self.log_text['path'].append(f'{filepath.absolute()}')

    def save_log(self):
        cls_name = self.__class__.__name__
        filename = f'log/log_{cls_name}_{datetime.datetime.now()}.csv'
        filename = filename.replace(':',
                                    '-')
        df_log_text = pd.DataFrame(self.log_text)
        df_log_text.to_csv(filename, encoding='utf-8')
