import os
from pathlib import Path

from config.config import Config
from lib.assets.autodict import AutoDict
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import count_decoding, decode_video


class MakeDectime(Worker, DectimePaths):
    total: int
    stdout: str
    items: AutoDict
    t: ProgressBar

    def init(self):
        self.total = len(self.name_list) * len(self.projection_list) * 181 * len(self.quality_list) * len(self.chunk_list)

    def main(self):
        for self.attempt in range(self.config.decoding_num):
            self.decode_chunks()

    def decode_chunks(self):
        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            with task(self):
                print(f'{self.ctx}', end='')
                self.count_dectime()
                self.check_decodable()
                self.decode()
                self.save()

    def count_dectime(self):
        try:
            self.turn = count_decoding(self.dectime_log)
            if self.turn >= self.decoding_num:
                raise AbortError('Decoding is enough.')
        except FileNotFoundError:
            return
        except UnicodeDecodeError:
            msg = 'UnicodeDecodeError: dectime_log'
            self.logger.register_log(msg, self.decodable_chunk)
            self.dectime_log.unlink(missing_ok=True)

    def check_decodable(self):
        if not self.decodable_chunk.exists():
            msg = 'FileNotFoundError: decodable_chunk not exist'
            self.logger.register_log(msg, self.decodable_chunk)
            print('')
            raise AbortError(msg)

    def decode(self):
        self.stdout = decode_video(self.decodable_chunk,
                                   threads=1, ui_prefix='\n\t')

    def save(self):
        self.dectime_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dectime_log, 'a') as f:
            f.write('\n' + self.stdout)


if __name__ == '__main__':
    os.chdir('../')

    # config_file = Path('config/config_cmp_qp.json')
    # videos_file = Path('config/videos_reduced.json')

    config_file = Path('config/config_pres_qp.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    MakeDectime(ctx).run()
