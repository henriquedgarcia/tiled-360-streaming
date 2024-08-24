from lib.utils.segmenter_utils import create_segments
from .assets.context import ctx
from .assets.worker import Worker


class Segmenter(Worker):
    def main(self):
        ctx.quality_list = ['0'] + ctx.quality_list
        create_segments()
