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
        dash_mpd = self.decodable_paths.dash_mpd.stat().st_size
        dash_init = {}
        dash_m4s = AutoDict()
        for self.quality in self.quality_list:
            dash_init[self.quality] = self.decodable_paths.dash_init.stat().st_size
            for self.chunk in self.chunk_list:
                self.t.set_postfix_str(f'{self.ctx}')
                self.t.update()
                dash_m4s[self.quality][self.chunk] = self.decodable_paths.dash_m4s.stat().st_size

        result_dict = {'dash_mpd': dash_mpd,
                       'dash_init': dash_init,  # dash_init[quality]
                       'dash_m4s': dash_m4s,  # dash_m4s[quality][chunk]
                       }
        self.set_bitrate(result_dict)
        self.quality = self.chunk = None

    def get_bitrate(self, file_tipe: str):
        """
        self.video_bitrate[name][projection][tiling][tile]
        :param file_tipe: ['dash_mpd'] |
                          ['dash_init'][self.quality] |
                          ['dash_m4s'][self.quality][self.chunk]
        :return:
        """
        keys = [self.name, self.projection, self.tiling, self.tile]
        if file_tipe == 'dash_init':
            keys.extend(['dash_init', self.quality])
        if file_tipe == 'dash_m4s':
            keys.extend(['dash_m4s', self.quality, self.chunk])
        return get_nested_value(self.video_bitrate, keys)

    def set_bitrate(self, value: dict):
        keys = [self.name, self.projection, self.tiling, self.tile]
        get_nested_value(self.video_bitrate, keys).update(value)
