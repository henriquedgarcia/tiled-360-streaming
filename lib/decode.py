from lib.assets.errors import AbortError
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.worker_utils import decode_video, count_decoding, print_error


class Decode(Worker):
    dectime_paths: DectimePaths
    changes: bool
    stdout: str
    items = []

    def main(self):
        self.dectime_paths = DectimePaths(context=self.ctx)
        self.decode_chunks()

    def decode_chunks(self):
        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            self.items.append((self.name, self.projection, self.tiling,
                               self.tile, self.quality, self.chunk))

        for self.attempt in range(self.config.decoding_num):
            for item in self.items:
                (self.name, self.projection, self.tiling,
                 self.tile, self.quality, self.chunk) = item

                with task(self):
                    self.work()
                    if self.turn >= 5:
                        self.items.remove(item)

    def work(self):
        self.check_dectime()
        self.check_decodable()
        self.decode()
        self.save()

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
        msg = []
        if not self.decodable_chunk.exists():
            msg += ['decodable_chunk not exist']
            self.logger.register_log('decodable_chunk not found.', self.decodable_chunk)
        if msg:
            raise AbortError('/'.join(msg))

    def decode(self):
        self.stdout = decode_video(self.decodable_chunk, threads=1, ui_prefix='\t')

    def save(self):
        self.dectime_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dectime_log, 'a') as f:
            f.write('\n' + self.stdout)

        self.changes = True

    @property
    def dectime_log(self):
        return self.dectime_paths.dectime_log

    @property
    def decodable_chunk(self):
        return self.dectime_paths.decodable_chunk

    @property
    def dectime_folder(self):
        return self.dectime_paths.dectime_folder
