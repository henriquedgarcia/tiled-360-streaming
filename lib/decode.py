from lib.assets.context import ctx
from lib.assets.worker import Worker
from lib.utils.decode_utils import make_decode


class Decode(Worker):
    turn: int

    def main(self):
        for ctx.name in ctx.name_list:
            for ctx.projection in ctx.name_list:
                for ctx.quality in ctx.quality_list:
                    for ctx.tiling in ctx.tiling_list:
                        for ctx.tile in ctx.tile_list:
                            for ctx.chunk in ctx.chunk_list:
                                make_decode()
