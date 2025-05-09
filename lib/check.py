from datetime import datetime

import pandas as pd
from tqdm import tqdm

from lib.assets.paths.make_chunk_quality_paths import MakeChunkQualityPaths
from lib.assets.worker import Worker


class Check(Worker, MakeChunkQualityPaths):
    def main(self):
        func = [self.CheckChunkQuality, self.CheckMakeDecodable, self.CheckMakeDash]
        options = ''.join(f'{n} - {c.__name__}\n' for n, c in enumerate(func))
        print(options)
        n = input('Choose option: ')
        func[int(n)]()

    def CheckChunkQuality(self):
        check_data = []
        total = len(self.name_list) * 181 * len(self.projection_list) * len(self.quality_list) * len(self.chunk_list)
        n = iter(range(total))
        columns = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'err']

        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        for self.quality in self.quality_list:
                            for self.chunk in self.chunk_list:
                                context = (f'{self.name}', f'{self.projection}', f'{self.tiling}', f'tile{self.tile}',
                                           f'qp{self.quality}', f'chunk{self.chunk}')
                                context_str = '_'.join(context)
                                msg = f'{next(n)}/{total} - {self.__class__.__name__} - {context_str}'
                                print(f'\r{msg}', end='')
                                err = ''
                                try:
                                    size = self.chunk_quality_json.stat().st_size
                                    if size == 0:
                                        err = 'size == 0'
                                except FileNotFoundError:
                                    err = 'FileNotFoundError'
                                if err:
                                    check_data.append(context + (err,))
        df = pd.DataFrame(check_data, columns=columns)
        now = f'{datetime.now()}'.replace(':', '-')
        df.to_csv(f'check_{self.__name__}_{now}.csv')
        pd.options.display.max_columns = len(df.columns)
        print(df)

    def CheckMakeDecodable(self):
        name = 'CheckMakeDecodable'
        check_data = []
        total = len(self.name_list) * 181 * len(self.projection_list) * len(self.quality_list) * len(self.chunk_list)
        columns = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'err']
        bar = tqdm(total=total, desc=name)
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        for self.quality in self.quality_list:
                            for self.chunk in self.chunk_list:
                                context = (f'{self.name}', f'{self.projection}', f'{self.tiling}', f'tile{self.tile}',
                                           f'qp{self.quality}', f'chunk{self.chunk}')
                                context_str = '_'.join(context)
                                bar.update()
                                bar.set_postfix_str(context_str)
                                err = ''
                                try:
                                    size = self.decodable_chunk.stat().st_size
                                    if size == 0:
                                        err = 'size == 0'
                                except FileNotFoundError:
                                    err = 'FileNotFoundError'
                                if err:
                                    check_data.append(context + (err,))

        df = pd.DataFrame(check_data, columns=columns)
        now = f'{datetime.now()}'.replace(':', '-')
        df.to_csv(f'check_{name}_{now}.csv')
        pd.options.display.max_columns = len(df.columns)
        print(df)

    def CheckMakeDash(self):
        name = 'CheckMakeDash'
        check_data = []
        total = len(self.name_list) * 181 * len(self.projection_list) * len(self.quality_list) * len(self.chunk_list)
        columns = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'err']
        bar = tqdm(total=total, desc=name)
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        err = ''
                        try:
                            size = self.dash_mpd.stat().st_size
                            if size == 0:
                                err = 'dash_mpd size == 0'
                        except FileNotFoundError:
                            err = 'dash_mpd FileNotFoundError'
                        if err:
                            context = (f'{self.name}', f'{self.projection}', f'{self.tiling}', f'tile{self.tile}',
                                       None, None)
                            check_data.append(context + (err,))

                        for self.quality in self.quality_list:
                            err = ''
                            try:
                                size = self.dash_init.stat().st_size
                                if size == 0:
                                    err = 'dash_init size == 0'
                            except FileNotFoundError:
                                err = 'dash_init FileNotFoundError'
                            if err:
                                context = (f'{self.name}', f'{self.projection}', f'{self.tiling}', f'tile{self.tile}',
                                           f'qp{self.quality}', None)
                                check_data.append(context + (err,))

                            for self.chunk in self.chunk_list:
                                bar.update()
                                context = (f'{self.name}', f'{self.projection}', f'{self.tiling}', f'tile{self.tile}',
                                           f'qp{self.quality}', f'chunk{self.chunk}')
                                bar.set_postfix_str('_'.join(context))
                                err = ''

                                try:
                                    size = self.dash_m4s.stat().st_size
                                    if size == 0:
                                        err = 'dash_m4s size == 0'
                                except FileNotFoundError:
                                    err = 'dash_m4s FileNotFoundError'

                                if err:
                                    check_data.append(context + (err,))

        df = pd.DataFrame(check_data, columns=columns)
        now = f'{datetime.now()}'.replace(':', '-')
        df.to_csv(f'check_{name}_{now}.csv')
        pd.options.display.max_columns = len(df.columns)
        print(df)
