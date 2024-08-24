from typing import Any

from lib.assets.context import ctx
from lib.assets.paths import paths
from lib.assets.worker import Worker
from lib.utils.decode_utils import skip_decode
from lib.utils.util import decode_file


def decode() -> Any:
    if skip_decode(): return

    print(f'==== Decoding {paths.segment_video} - Turn {ctx.turn + 1} ====')

    stdout = decode_file(paths.segment_video, threads=1)
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
