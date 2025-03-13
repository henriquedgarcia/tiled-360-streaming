import pandas as pd
from tqdm import tqdm

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.makedash import MakeDashPaths
from lib.utils.context_utils import task
from lib.utils.util import print_error, save_json, get_nested_value, save_pickle


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

    def iter_name_proj_tiling_tile_qlt_chunk(self):
        self.projection = 'cmp'
        self.progress_bar = ProgressBar(total=(len(self.name_list)
                                               * 181
                                               ),
                                        desc=f'{self.__class__.__name__}')
        for self.name in self.name_list:
            for self.tiling in self.tiling_list:
                for self.tile in self.tile_list:
                    self.progress_bar.update(f'{self.ctx}')
                    for self.quality in self.quality_list:
                        for self.chunk in self.chunk_list:
                            yield

    def init(self):
        self.decodable_paths = MakeDashPaths(self.ctx)
        if self.decodable_paths.bitrate_result_pickle.exists():
            print('file exists')
            exit(0)

    def main(self):
        bitrate_result = []
        for _ in self.iter_name_proj_tiling_tile_qlt_chunk():
            bitrate = self.decodable_paths.dash_m4s.stat().st_size * 8
            bitrate_result.append((self.name, self.projection, self.tiling, int(self.tile), int(self.quality), int(self.chunk), bitrate))

        result = pd.DataFrame(bitrate_result, columns=['name', 'projection', 'tiling', 'tile', 'quality', 'chunk', 'bitrate'])
        result.set_index(['name', 'projection', 'tiling', 'tile', 'quality', 'chunk'], inplace=True)
        save_pickle(result['bitrate'], self.decodable_paths.bitrate_result_pickle)
        print('finished')
