import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config.config import Config
from lib.assets.context import Context
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.paths.makesitipaths import MakeSitiPaths
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import get_times


class Check(Worker, ViewportQualityPaths, DectimePaths, MakeSitiPaths):
    def main(self):
        func = [self.CheckMakeDash, self.CheckMakeDecodable, self.CheckMakeDectime,
                self.CheckChunkQuality, self.CheckTilesSeen, self.CheckViewportQuality, self.CheckMakeSITI]
        options = ''.join(f'{n} - {c.__name__}\n' for n, c in enumerate(func))
        print(options)
        n = input('Choose option: ')
        func[int(n)]()

    def CheckMakeTiles(self):
        name = 'CheckMakeTiles'
        check_data = []
        total = len(self.name_list) * 181 * len(self.projection_list) * len(self.quality_list) * len(self.chunk_list)
        columns = ['name', 'projection', 'tiling', 'tile', 'quality', 'err']
        bar = tqdm(total=total, desc=name)
        for _ in self.iterate_name_projection_tiling_tile_quality:
            context = (f'{self.name}', f'{self.projection}', f'{self.tiling}', f'tile{self.tile}',
                       f'qp{self.quality}', f'chunk{self.chunk}')
            context_str = '_'.join(context)

            bar.update()
            bar.set_postfix_str(context_str)

            err = ''
            try:
                size = self.tile_video.stat().st_size
                if size == 0:
                    err = 'VideoSize==0'
            except FileNotFoundError:
                err = 'VideoFileNotFoundError'

            try:
                compressed_log_text = self.tile_log.read_text()
                if 'encoded 1800 frames' not in compressed_log_text:
                    err += '_EncoderError'
                if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
                    err += '_CodecVersionError'
            except FileNotFoundError:
                err = 'LogFileNotFoundError'

            if err:
                check_data.append(context + (err,))

        df = pd.DataFrame(check_data, columns=columns)
        now = f'{datetime.now()}'.replace(':', '-')
        df.to_csv(f'check_{name}_{now}.csv')
        pd.options.display.max_columns = len(df.columns)
        print(df)

    def CheckChunkQuality(self):
        check_data = []
        total = len(self.name_list) * 181 * len(self.projection_list) * len(self.quality_list) * len(self.chunk_list)
        n = iter(range(total))
        columns = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'err']

        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
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
        df.to_csv(f'{self.CheckChunkQuality.__name__}_{now}.csv')
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

    def CheckViewportQuality(self):
        name = 'CheckMakeDectime'
        check_data = []
        total = len(self.name_list) * len(self.tiling_list) * len(self.projection_list) * len(self.quality_list) * 30 * len(self.chunk_list)
        columns = ['name', 'projection', 'tiling', 'quality', 'user', 'chunk', 'err']
        bar = ProgressBar(total=total, desc=name)

        for _ in self.iterate_name_projection_tiling_quality_user_chunk:
            context = (f'{self.name}', f'{self.projection}', f'{self.tiling}',
                       f'qp{self.quality}', f'user{self.user}', f'chunk{self.chunk}')
            bar.update('_'.join(context))

            if not self.user_viewport_quality_json.exists():
                err = 'FileNotFoundError'
                check_data.append(context + (err,))

        df = pd.DataFrame(check_data, columns=columns)
        now = f'{datetime.now()}'.replace(':', '-')
        df.to_csv(f'check_{name}_{now}.csv')
        pd.options.display.max_columns = len(df.columns)
        print(df)

    def CheckMakeDectime(self):
        name = 'CheckMakeDectime'
        check_data = []
        total = len(self.name_list) * 181 * len(self.projection_list) * len(self.quality_list) * len(self.chunk_list)
        columns = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'err']
        bar = ProgressBar(total=total, desc=name)
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        for self.quality in self.quality_list:
                            for self.chunk in self.chunk_list:
                                context = (f'{self.name}', f'{self.projection}', f'{self.tiling}', f'tile{self.tile}',
                                           f'qp{self.quality}', f'chunk{self.chunk}')
                                context_str = '_'.join(context)
                                bar.update(context_str)

                                if not self.dectime_log.exists():
                                    err = 'FileNotFoundError'
                                    check_data.append(context + (err,))
                                    continue

                                dectime_list = get_times(self.dectime_log)
                                decoding_times = len(dectime_list)

                                if decoding_times < self.decoding_num:
                                    err = 'DecodeTimesError'
                                    check_data.append(context + (err,))
                                    continue

                                if 0 in dectime_list:
                                    err = '0InDectimeError'
                                    check_data.append(context + (err,))
                                    continue

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

    def CheckMakeSITI(self):
        name = 'CheckMakeSITI'
        check_data = []
        total = len(self.name_list) * 181 * len(self.projection_list) * len(self.quality_list) * len(self.chunk_list)
        columns = ['name', 'projection', 'tiling', 'tile', 'quality', 'err']
        bar = tqdm(total=total, desc=name)

        self.projection = 'cmp'
        self.tiling = '3x2'
        self.quality = '28'

        for self.name in self.name_list:
            for self.tile in self.tile_list:
                bar.update()
                context = (f'{self.name}', f'{self.projection}', f'{self.tiling}', f'tile{self.tile}',
                           f'qp{self.quality}')
                bar.set_postfix_str('_'.join(context))

                err = ''
                try:
                    size = pd.read_csv(self.siti_csv_results, index_col=0).size
                    if size != 3600:
                        err = 'siti_csv_results size != 3600'
                except FileNotFoundError:
                    err = 'siti_csv_results FileNotFoundError'
                if err:
                    check_data.append(context + (err,))

        df = pd.DataFrame(check_data, columns=columns)
        now = f'{datetime.now()}'.replace(':', '-')
        df.to_csv(f'check_{name}_{now}.csv')
        pd.options.display.max_columns = len(df.columns)
        print(df)

    def CheckTilesSeen(self):
        check_data = []
        total = len(self.name_list) * len(self.projection_list) * len(self.tiling_list) * 30
        n = iter(range(total))
        columns = ['name', 'projection', 'tiling', 'user', 'err']

        for _ in self.iterate_name_projection_tiling_user():
            context = (f'{self.name}', f'{self.projection}', f'{self.tiling}',
                       f'user{self.user}')
            context_str = '_'.join(context)
            msg = f'{next(n) + 1}/{total} - {self.CheckTilesSeen.__name__} - {context_str}'
            print(f'\r{msg}', end='')

            err = ''
            try:
                size = self.user_seen_tiles_json.stat().st_size
                if size < 10:
                    err = 'size < 10'
            except FileNotFoundError:
                err = 'FileNotFoundError'
            if err:
                check_data.append(context + (err,))

        df = pd.DataFrame(check_data, columns=columns)
        now = f'{datetime.now()}'.replace(':', '-')
        df.to_csv(f'{self.CheckTilesSeen.__name__}_{now}.csv')
        pd.options.display.max_columns = len(df.columns)
        print('')
        print(df)


if __name__ == '__main__':
    os.chdir('../')

    # config_file = Path('config/config_cmp_qp.json')
    # videos_file = Path('config/videos_reduced.json')

    config_file = Path('config/config_pres_qp.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    Check(ctx).run()
