import pandas as pd
from pandas import MultiIndex

from lib.assets.ctxinterface import CtxInterface
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.make_dash import MakeDashPaths


class GetBitrate(Worker, MakeDashPaths, CtxInterface):
    """
    self.video_bitrate[name][projection][tiling][tile]  |
                                  ['dash_mpd'] |
                                  ['dash_init'][self.quality] |
                                  ['dash_m4s'][self.quality][self.chunk]
    :return:
    """
    result_data: pd.DataFrame

    def init(self):
        self.projection = 'cmp'

    def main(self):
        self.data = []
        for self.name in self.name_list:
            for self.tiling in self.tiling_list:
                for self.tile in self.tile_list:
                    print(f'{self.__class__.__name__} - {self.name}_{self.tiling}_tile{self.tile}')
                    for self.quality in self.quality_list:
                        for self.chunk in self.chunk_list:
                            bitrate = self.dash_m4s.stat().st_size * 8
                            key = (self.name, self.projection, self.tiling,
                                   int(self.tile), int(self.quality),
                                   int(self.chunk) - 1, bitrate)
                            self.data.append(key)

        print('Saving Pickle')
        cools_names = ['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'bitrate']
        df = pd.DataFrame(self.data, columns=cools_names)
        df.set_index(cools_names[:-1], inplace=True)
        df.sort_index(inplace=True)
        pd.to_pickle(df, self.bitrate_result_pickle)
        print('finished')
