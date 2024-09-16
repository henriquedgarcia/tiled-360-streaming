from lib.utils.segmenter_utils import create_segments
from .assets.worker import Worker


class Segmenter(Worker):
    def main(self):
        self.ctx.quality_list = ['0'] + self.ctx.quality_list
        create_segments()
