from lib.assets.worker import Worker
from lib.utils.decode_utils import decode_chunks


class Decode(Worker):
    turn: int

    def main(self):
        decode_chunks()
