from lib.assets.worker import Worker
from lib.utils.decode_utils import make_decode


class Decode(Worker):
    turn: int

    def main(self):
        make_decode()
