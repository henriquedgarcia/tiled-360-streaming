from typing import Any

from lib.assets.autodict import AutoDict
from lib.assets.worker import Worker
from lib.utils.worker_utils import save_json, load_json, print_error
from lib.segmenter import Segmenter, SegmenterPaths
from lib.assets.ctxinterface import CtxInterface
from lib.utils.worker_utils import get_nested_value

class GetBitrate(Segmenter, CtxInterface):
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

    def main(self):
        self.segmenter_paths = SegmenterPaths(self.config, self.ctx)
        self.ctx.quality_list = ['0'] + self.ctx.quality_list
        self.collect_bitrate()

    def collect_bitrate(self):
        for self.name in self.name_list:
            if self.segmenter_paths.bitrate_result_json.exists():
                print_error(f'\tThe bitrate_result_json exist.')
                continue

            self.video_bitrate = AutoDict()

            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                self.bitrate()

            print(f'Saving video_bitrate for {self.name}.')
            save_json(self.video_bitrate,
                      self.segmenter_paths.bitrate_result_json)

    def bitrate(self):
        print(f'\t{self.ctx}', end='\r')

        try:
            chunk_size = self.segmenter_paths.chunk_video.stat().st_size
        except FileNotFoundError:
            chunk_size = 0

        bitrate = 8 * chunk_size / 1  # chunk duration is 1 second

        if chunk_size == 0:
            self.logger.register_log('BITRATE==0',
                                     self.segmenter_paths.chunk_video)

        self.set_bitrate({'bitrate': bitrate,
                          })

    def get_bitrate(self):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        return get_nested_value(self.video_bitrate, keys)

    def set_bitrate(self, value: dict):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        get_nested_value(self.video_bitrate, keys).update(value)
