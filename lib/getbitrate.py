import pandas as pd

from lib.assets.ctxinterface import CtxInterface
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.makedash import MakeDashPaths
from lib.utils.util import save_pickle


class GetBitrate(Worker, CtxInterface):
    """
    self.video_bitrate[name][projection][tiling][tile]  |
                                  ['dash_mpd'] |
                                  ['dash_init'][self.quality] |
                                  ['dash_m4s'][self.quality][self.chunk]
    :return:
    """
    decodable_paths: MakeDashPaths
    progress_bar: ProgressBar

    def iter_proj_tiling_tile_qlt_chunk(self):
        self.projection = 'cmp'
        self.progress_bar = ProgressBar(total=(len(self.quality_list)
                                               * 181
                                               ),
                                        desc=f'{self.__class__.__name__}')
        for self.tiling in self.tiling_list:
            for self.tile in self.tile_list:
                self.progress_bar.update(f'{self.ctx}')
                for self.quality in self.quality_list:
                    for self.chunk in self.chunk_list:
                        yield

    def init(self):
        self.decodable_paths = MakeDashPaths(self.ctx)

    def main(self):
        for self.name in self.name_list:
            if self.decodable_paths.bitrate_result_pickle.exists():
                print('file exists')
                continue

            bitrate_result = []
            for _ in self.iter_proj_tiling_tile_qlt_chunk():
                bitrate = self.get_bitrate()
                self.set_bitrate(bitrate_result, bitrate)

            result = pd.DataFrame(bitrate_result,
                                  columns=['name', 'projection',
                                           'tiling', 'tile',
                                           'quality', 'chunk',
                                           'bitrate'])
            result.set_index(['name', 'projection', 'tiling',
                              'tile', 'quality', 'chunk'],
                             inplace=True)
            save_pickle(result['bitrate'],
                        self.decodable_paths.bitrate_result_pickle)
            print('finished')

    def set_bitrate(self, bitrate_result, bitrate):
        key = [self.name, self.projection, self.tiling,
               int(self.tile), int(self.quality),
               int(self.chunk) - 1, bitrate]
        bitrate_result.append(key)

    def get_bitrate(self):
        chunk_size = self.decodable_paths.dash_m4s.stat().st_size
        return chunk_size * 8
