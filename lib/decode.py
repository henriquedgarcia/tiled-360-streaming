from typing import Any

from lib.assets import Worker, ctx, paths
from lib.assets import config
from lib.utils import decode_file, get_times, print_error


def clean_dectime_log():
    paths.dectime_log.unlink(missing_ok=True)


class Decode(Worker):
    turn: int

    def main(self):
        for ctx.name in ctx.name_list:
            for ctx.projection in ctx.name_list:
                for ctx.quality in ctx.quality_list:
                    for ctx.tiling in ctx.tiling_list:
                        for ctx.tile in ctx.tile_list:
                            for ctx.chunk in ctx.chunk_list:
                                self.decode()

    def decode(self) -> Any:
        if self.skip_decode(): return

        print(f'Decoding file "{paths.segment_file}". ', end='')

        print(f'Turn {ctx.turn + 1}')
        stdout = decode_file(paths.segment_file, threads=1)
        with paths.dectime_log.open('a') as f:
            f.write(f'\n==========\n{stdout}')
            print(' OK')

    def skip_decode(self):
        # ctx.turn = 0

        try:
            content = paths.dectime_log.read_text(encoding='utf-8')
            times = get_times(content)
            self.turn = len(times)
            if self.turn < config.decoding_num:
                raise FileNotFoundError
            print(f' Decoded {self.turn}.')
            return True
        except FileNotFoundError:
            if self.segment_file.exists():
                return False
            else:
                print_error(f'The segment not exist. ')
                self.log("segment_file not exist.",
                         self.segment_file)
                return True
