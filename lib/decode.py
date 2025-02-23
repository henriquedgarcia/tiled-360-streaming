from contextlib import contextmanager

from lib.assets.autodict import AutoDict
from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import count_decoding, decode_video


class Decode(Worker):
    dectime_paths: DectimePaths
    total: int
    stdout: str
    items: AutoDict
    t: ProgressBar

    def main(self):
        self.init()
        for self.attempt in range(self.config.decoding_num):
            self.decode_chunks()

    def init(self):
        self.dectime_paths = DectimePaths(context=self.ctx)

    def decode_chunks(self):
        for _ in self.iter_items():
            if self.status.get_status('is_ok'):
                continue

            with self.task():
                self.check_dectime()
                self.check_decodable()
                self.stdout = decode_video(self.decodable_chunk, threads=1, ui_prefix='\t')

    def iter_items(self):
        total = len(self.name_list) * len(self.projection_list) * 181 * len(self.quality_list) * len(self.chunk_list)
        self.t = ProgressBar(total=total, desc=self.__class__.__name__)
        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            self.t.update(f'{self.ctx}')
            yield

    def check_dectime(self):
        try:
            self.turn = count_decoding(self.dectime_log)
        except (FileNotFoundError, UnicodeDecodeError):
            self.turn = 0
            self.dectime_log.unlink(missing_ok=True)
        if self.turn < self.decoding_num:
            return
        self.status.update_status('is_ok', True)
        raise AbortError()

    def check_decodable(self):
        if not self.decodable_chunk.exists():
            msg = 'decodable_chunk not exist'
            self.logger.register_log(msg, self.decodable_chunk)
            raise AbortError()

    @contextmanager
    def task(self):
        try:
            yield
        except AbortError:
            return
        self.save()

    def save(self):
        self.dectime_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dectime_log, 'a') as f:
            f.write('\n' + self.stdout)

    @property
    def dectime_log(self):
        return self.dectime_paths.dectime_log

    @property
    def decodable_chunk(self):
        return self.dectime_paths.decodable_chunk

    @property
    def dectime_folder(self):
        return self.dectime_paths.dectime_folder
