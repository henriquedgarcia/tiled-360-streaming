from contextlib import contextmanager

from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.worker import Worker, ProgressBar
from lib.utils.context_utils import task
from lib.utils.worker_utils import decode_video, count_decoding, print_error


class Decode(Worker):
    dectime_paths: DectimePaths
    stdout: str
    items = []

    def iter_items(self):
        t = ProgressBar(total=len(self.items), desc=self.__class__.__name__)
        for item in self.items:
            t.update(f'{self.ctx}')
            (self.name, self.projection, self.tiling,
             self.tile, self.quality, self.chunk) = item
            yield

            if self.turn >= 5:
                self.items.remove(item)

    def main(self):
        self.init()
        for self.attempt in range(self.config.decoding_num):
            self.decode_chunks()

    def decode_chunks(self):
        for _ in self.iter_items():
            with task(self, verbose=False):
                self.stdout = decode_video(self.decodable_chunk, threads=1, ui_prefix='\t')
                self.save()

    def init(self):
        self.dectime_paths = DectimePaths(context=self.ctx)
        self.make_task_list()

    @contextmanager
    def task(self):
        print(f'==== {self.__class__.__name__} {self.ctx} ====')

        try:
            yield
        except AbortError as e:
            print_error(f'\t{e.args[0]}')
            return

        self.save()

    def save(self):
        self.dectime_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dectime_log, 'a') as f:
            f.write('\n' + self.stdout)

    def iterator(self):
        total = len(self.name_list) * len(self.projection_list) * 181 * len(self.quality_list) * len(self.chunk_list)
        t = ProgressBar(total=total, desc=self.__class__.__name__)

        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            t.update(f'{self.ctx}')
            yield

    def check_dectime(self):
        try:
            self.turn = count_decoding(self.dectime_log)
        except (FileNotFoundError, UnicodeDecodeError):
            self.turn = 0
            self.dectime_log.unlink(missing_ok=True)
        if self.turn < self.decoding_num:
            print_error(f'\tDecoded {self.turn} times.')
            return
        raise AbortError(f'Dectime is OK. {self.turn}/{self.decoding_num} times. Skipping.')

    def check_decodable(self):
        if not self.decodable_chunk.exists():
            msg = 'decodable_chunk not exist'
            self.logger.register_log(msg, self.decodable_chunk)
            raise AbortError('/'.join(msg))

    def make_task_list(self):
        for _ in self.iterator():
            with self.task():
                self.check_dectime()
                self.check_decodable()
                self.items.append((self.name, self.projection, self.tiling,
                                   self.tile, self.quality, self.chunk))

    @property
    def dectime_log(self):
        return self.dectime_paths.dectime_log

    @property
    def decodable_chunk(self):
        return self.dectime_paths.decodable_chunk

    @property
    def dectime_folder(self):
        return self.dectime_paths.dectime_folder
