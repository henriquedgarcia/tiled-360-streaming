from tqdm import tqdm

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.worker import Worker
from lib.makedash import MakeDashPaths
from lib.utils.context_utils import task
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
    t: tqdm

    def main(self):
        self.decodable_paths = MakeDashPaths(self.ctx)

        for self.name in self.name_list:
            print(f'==== {self.__class__.__name__} {self.name} ====')
            if self.decodable_paths.bitrate_result_json.exists():
                print_error(f'\tThe bitrate_result_json exist.')
                continue

            self.video_bitrate = AutoDict()
            self.t = tqdm(total=(181
                                 * len(self.projection_list)
                                 * len(self.quality_list)
                                 * len(self.chunk_list)),
                          desc=f'    {self.__class__.__name__}')
            for _ in self.iterate_projection_tiling_tile():
                with task(self, verbose=False):
                    self.work()
            self.t.close()

            print(f'\tSaving video_bitrate for {self.name}.')
            save_json(self.video_bitrate,
                      self.decodable_paths.bitrate_result_json)

    def work(self):
        bitrate = AutoDict()

        for self.quality in self.quality_list:
            for self.chunk in self.chunk_list:
                self.t.set_postfix_str(f'{self.ctx}')
                self.t.update()
                bitrate[self.quality][self.chunk] = self.decodable_paths.dash_m4s.stat().st_size
            bitrate[self.quality]['dash_init'] = self.decodable_paths.dash_init.stat().st_size
        bitrate['dash_mpd'] = self.decodable_paths.dash_mpd.stat().st_size

        self.set_bitrate(bitrate)
        self.quality = self.chunk = None

    def set_bitrate(self, value: dict):
        keys = [self.name, self.projection, self.tiling, self.tile]
        tile_bitrate = get_nested_value(self.video_bitrate, keys)
        tile_bitrate.update(value)
