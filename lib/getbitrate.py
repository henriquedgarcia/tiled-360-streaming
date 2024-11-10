from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.worker import Worker
from lib.makedash import MakeDashPaths
from lib.utils.context_utils import task
from lib.utils.worker_utils import get_nested_value
from lib.utils.worker_utils import save_json, print_error


class GetBitrate(Worker, CtxInterface):
    """
       The result dict have a following structure:
       results[video_name][tile_pattern][quality][tile_id][chunk_id]
               ['times'|'rate']
       [video_proj]    : The video projection
       [video_name]    : The video name
       [tile_pattern]  : The tile tiling. e.g. "6x4"
       [quality]       : Quality. An int like in crf or qp.
       [tile_id]           : the tile number. ex. max = 6*4
       [chunk_id]           : the chunk number. Start with 1.

    """
    video_bitrate: AutoDict
    quality_list: list
    decodable_paths: MakeDashPaths

    def main(self):
        self.init()
        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            with task(self):
                self.work()

    def init(self):
        self.decodable_paths = MakeDashPaths(self.ctx)
        self.quality_list = ['0'] + self.ctx.quality_list
        self.collect_bitrate()

    def work(self):
        self.collect_bitrate()

    def collect_bitrate(self):
        for self.name in self.name_list:
            if self.decodable_paths.bitrate_result_json.exists():
                print_error(f'\tThe bitrate_result_json exist.')
                continue

            self.video_bitrate = AutoDict()

            with task(self):
                for _ in self.iterate_projection_tiling_tile_quality_chunk():
                    self.bitrate()

            print(f'Saving video_bitrate for {self.name}.')
            save_json(self.video_bitrate,
                      self.decodable_paths.bitrate_result_json)

    def bitrate(self):
        print(f'\t{self.ctx}', end='\r')

        try:
            dash_m4s = self.decodable_paths.dash_m4s.stat().st_size
            bitrate = 8 * dash_m4s
            if dash_m4s == 0:
                self.logger.register_log('BITRATE==0', self.decodable_paths.dash_m4s)

        except FileNotFoundError:
            self.logger.register_log('chunk not exist', self.decodable_paths.dash_m4s)
            raise AbortError('Chunk not exist')

        self.set_bitrate({'bitrate': bitrate, })

    def get_bitrate(self):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        return get_nested_value(self.video_bitrate, keys)

    def set_bitrate(self, value: dict):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        get_nested_value(self.video_bitrate, keys).update(value)
