from contextlib import contextmanager

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.worker import Worker, ProgressBar
from lib.makedash import MakeDashPaths
from lib.utils.worker_utils import get_nested_value, save_json, print_error


class GetBitrate(Worker, CtxInterface):
    """
    self.video_bitrate[name][projection][tiling][tile]  |
                                  ['dash_mpd'] |
                                  ['dash_init'][self.quality] |
                                  ['dash_m4s'][self.quality][self.chunk]
    :return:
    """
    video_bitrate: AutoDict
    decodable_paths: MakeDashPaths
    total: int
    t: ProgressBar

    def main(self):
        self.init()

        for self.name in self.name_list:
            with self.task():
                for _ in self.iterator():
                    self.work()

    def init(self):
        self.decodable_paths = MakeDashPaths(self.ctx)
        self.total = (181
                      * len(self.projection_list)
                      * len(self.quality_list)
                      * len(self.chunk_list))

    def iterator(self):
        self.t = ProgressBar(total=self.total, desc=self.__class__.__name__)
        for self.projection in self.projection_list:
            for self.tiling in self.tiling_list:
                for self.tile in self.tile_list:
                    self.ctx.iterations += 1
                    yield
        self.t.close()

    @contextmanager
    def task(self):
        print(f'==== {self.__class__.__name__} {self.name} ====')

        self.video_bitrate = AutoDict()
        try:
            if self.decodable_paths.bitrate_result_json.exists():
                raise AbortError(f'The bitrate_result_json exist.')
            yield
        except AbortError as e:
            print_error(f'\t{e.args[0]}')
            return

        print(f'\tSaving for {self.name}.')
        save_json(self.video_bitrate,
                  self.decodable_paths.bitrate_result_json)

    def work(self):
        bitrate = AutoDict()

        for self.quality in self.quality_list:
            for self.chunk in self.chunk_list:
                self.t.update(f'{self.ctx}')
                bitrate[self.quality][self.chunk]['dash_m4s'] = self.decodable_paths.dash_m4s.stat().st_size
            bitrate[self.quality]['dash_init'] = self.decodable_paths.dash_init.stat().st_size
        bitrate['dash_mpd'] = self.decodable_paths.dash_mpd.stat().st_size

        self.set_bitrate(bitrate)
        self.quality = self.chunk = None

    def set_bitrate(self, value: dict):
        keys = [self.name, self.projection, self.tiling, self.tile]
        tile_bitrate = get_nested_value(self.video_bitrate, keys)
        tile_bitrate.update(value)
