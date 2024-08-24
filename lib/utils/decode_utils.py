from typing import Any

from config.config import config
from lib.assets.context import ctx
from lib.assets.logger import logger
from lib.assets.paths import paths
from lib.utils.segmenter_utils import segment
from lib.utils.util import get_times, decode_file


def make_decode():
    for ctx.turn in range(config.decoding_num):
        for _ in iter_decode():
            decode()


def iter_decode():
    for ctx.name in ctx.name_list:
        for ctx.projection in ctx.name_list:
            for ctx.quality in ctx.quality_list:
                for ctx.tiling in ctx.tiling_list:
                    for ctx.tile in ctx.tile_list:
                        for ctx.chunk in ctx.chunk_list:
                            yield


def decode():
    check_decode()
    if (logger.get_status('decode_ok')
            or not logger.get_status('segments_ok')):
        return

    print(f'==== Decoding {paths.segment_video} - Turn {ctx.turn} ====')

    stdout = decode_file(paths.segment_video, threads=1)
    with paths.dectime_log.open('a') as f:
        f.write(f'\n==========\n{stdout}')
    check_decode()


def check_decode():
    if not logger.get_status('decode_ok'):
        check_dectime_log()

    if not logger.get_status('segments_ok'):
        segment()


def clean_dectime_log():
    paths.dectime_log.unlink(missing_ok=True)


def check_dectime_log():
    try:
        dectime_log_content = paths.dectime_log.read_text(encoding='utf-8')
        turn = len(get_times(dectime_log_content))
        if ctx.turn < config.decoding_num:
            logger.update_status('decode_ok', True)
            return
    except FileNotFoundError:
        clean_dectime_log()
    logger.update_status('decode_ok', True)
    print(f'\tDecoded {ctx.turn}.')
