from typing import Any

from lib.assets.autodict import AutoDict
from lib.assets.worker import Worker
from lib.utils.util import save_json, print_error, load_json, get_times


class GetDectime(Worker):
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

       'times': list(float, float, float)
       'rate': float
    """
    result_times: AutoDict
    change_flag: bool
    check_result = True
    error: bool

    def main(self):
        for self.video in self.video_list:
            if list(self.video_list).index(self.video) < 38: continue
            self.result_times = AutoDict()
            self.change_flag = True
            self.error = False
            if self.skip1(): continue

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.dectime()

            if self.change_flag and not self.error:
                print('Saving.')
                save_json(self.result_times,
                          self.dectime_result_json)

    def dectime(self) -> Any:
        print(f'\r{self.state_str()} = ',
              end='')

        try:
            times = self.get_dectime()
        except FileNotFoundError:
            self.error = True
            self.log('DECTIME_FILE_NOT_FOUND',
                     self.dectime_log)
            print_error(f'\n\tThe dectime log not exist. Skipping.')
            return

        if self.check_result:
            old_value = self.result_times[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            if old_value == times:
                return
            elif not self.change_flag:
                self.change_flag = True

        self.result_times[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = times
        print(f'{times}',
              end='')

    def skip1(self):
        if self.dectime_result_json.exists():
            print(f'\n[{self.proj}][{self.video}] - The dectime_result_json exist.')
            if self.check_result:
                self.change_flag = False
                self.result_times = load_json(self.dectime_result_json,
                                              object_hook=AutoDict)
                return False
            return True
        return False

    def get_dectime(self) -> Any:
        content = self.dectime_log.read_text(encoding='utf-8')
        times = get_times(content)
        times = times[-self.decoding_num:]
        times = sorted(times)

        if len(times) < self.decoding_num:
            print_error(f'\n    The dectime is lower than {self.decoding_num}: ')
            self.log(f'DECTIME_NOT_DECODED_ENOUGH_{len(times)}',
                     self.dectime_log)

        if 0 in times:
            print_error(f'\n    0  found: ')
            self.log('DECTIME_ZERO_FOUND',
                     self.dectime_log)

        return times

    def skip2(self):
        if not self.dectime_log.exists():
            return True
        return False
