from typing import Any

from lib.assets.autodict import AutoDict
from lib.assets.worker import Worker
from lib.utils.worker_utils import save_json, load_json, print_error


class GetBitrate(Worker):
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
    turn: int
    _video: str
    result_rate: AutoDict
    error: bool
    change_flag: bool
    check_result = True

    def main(self):
        for self.video in self.video_list:
            if list(self.video_list).index(self.video) < 0: continue
            self.result_rate = AutoDict()
            self.change_flag = True
            self.error = False
            if self.skip1(): continue

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.bitrate()

            if self.change_flag and not self.error:
                print('Saving.')
                save_json(self.result_rate,
                          self.bitrate_result_json)

    def skip1(self, ):
        if self.bitrate_result_json.exists():
            print(f'\n[{self.proj}][{self.video}] - The bitrate_result_json exist.')
            if self.check_result:
                self.change_flag = False
                self.result_rate = load_json(self.bitrate_result_json,
                                             object_hook=AutoDict)
                return False
            return True
        return False

    def bitrate(self) -> Any:
        print(f'\r{self.state_str()}: ',
              end='')

        try:
            bitrate = self.get_bitrate()
        except FileNotFoundError:
            self.error = True
            self.log('BITRATE FILE NOT FOUND',
                     self.dectime_log)
            print_error(f'\n\n\tThe segment not exist. Skipping.')
            return

        if self.check_result:
            old_value = self.result_rate[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            if old_value == bitrate:
                return
            elif not self.change_flag:
                self.change_flag = True

        self.result_rate[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = bitrate
        print(f'{self.bitrate}',
              end='')

    def get_bitrate(self):
        chunk_size = self.segment_file.stat().st_size

        if chunk_size == 0:
            self.log('BITRATE==0',
                     self.segment_file)
            self.segment_file.unlink()
            raise FileNotFoundError

        bitrate = 8 * chunk_size / self.chunk_dur

        return bitrate
