from typing import Any

from lib.assets import Worker, ctx, paths
from lib.utils import decode_file
from lib.utils.decode_utils import skip_decode


def decode() -> Any:
    if skip_decode(): return

    print(f'==== Decoding {paths.segment_file} - Turn {ctx.turn + 1} ====')

    stdout = decode_file(paths.segment_file, threads=1)
    with paths.dectime_log.open('a') as f:
        f.write(f'\n==========\n{stdout}')


class Decode(Worker):
    turn: int

    def main(self):
        for ctx.name in ctx.name_list:
            for ctx.projection in ctx.name_list:
                for ctx.quality in ctx.quality_list:
                    for ctx.tiling in ctx.tiling_list:
                        for ctx.tile in ctx.tile_list:
                            for ctx.chunk in ctx.chunk_list:
                                decode()

