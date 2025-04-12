import pandas as pd

from lib.assets.ctxinterface import CtxInterface
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.makedash import MakeDashPaths


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

    def init(self):
        self.projection = 'cmp'

    def iter_proj_tiling_tile_qlt_chunk(self):
        self.progress_bar = ProgressBar(total=(len(self.quality_list) * 181),
                                        desc=f'{self.__class__.__name__}')
        for self.tiling in self.tiling_list:
            for self.tile in self.tile_list:
                self.progress_bar.update(f'{self.ctx}')
                for self.quality in self.quality_list:
                    for self.chunk in self.chunk_list:
                        yield

    def main(self):
        columns = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'bitrate']
        for self.name in self.name_list:
            if self.decodable_paths.bitrate_result_pickle.exists():
                print('file exists')
                continue

            bitrate_result = self.get_results()

            result = pd.DataFrame(bitrate_result, columns=columns)
            result.set_index(columns[:-1], inplace=True)
            result['bitrate'].to_pickle(self.decodable_paths.bitrate_result_pickle)
            print('finished')

    def get_results(self):
        bitrate_result = []
        self.progress_bar = ProgressBar(total=(len(self.quality_list) * 181),
                                        desc=f'{self.__class__.__name__}')

        for self.tiling in self.tiling_list:
            for self.tile in self.tile_list:
                self.progress_bar.update(f'{self.ctx}')
                for self.quality in self.quality_list:
                    for self.chunk in self.chunk_list:
                        bitrate = self.decodable_paths.dash_m4s.stat().st_size * 8
                        key = [self.name, self.projection, self.tiling,
                               int(self.tile), int(self.quality),
                               int(self.chunk) - 1, bitrate]
                        bitrate_result.append(key)
        return bitrate_result
