from config.config import config
from lib.assets.context import ctx
from lib.assets.logger import logger
from lib.assets.paths import segmenter_paths
from lib.utils.segmenter_utils import segment
from lib.utils.util import get_times, run_command, print_error


def make_decode():
    for ctx.turn in range(config.decoding_num):
        for _ in iter_decode():
            decode()


def decode():
    print(f'==== Decoding {ctx} ====')

    check_decode()

    print(f'\tTurn {ctx.turn}/{config.decoding_num}')

    if logger.get_status('decode_ok'):
        print('\tDectime is OK. Skipping')
    elif not logger.get_status('segments_ok'):
        print_error('\tChunk not exist.')
        return

    cmd = make_decode_cmd()
    run_command(cmd, segmenter_paths.dectime_log.parent, segmenter_paths.dectime_log, mode='a')

    check_decode()


def check_decode():
    if not logger.get_status('decode_ok'):
        check_dectime_log()

    if not logger.get_status('segments_ok'):
        segment()


def check_dectime_log():
    ctx.turn = get_turn()
    logger.update_status('decode_turn', ctx.turn)

    if ctx.turn == 0:
        clean_dectime_log()

    if logger.get_status('decode_turn') > config.decoding_num:
        logger.update_status('decode_ok', True)


def clean_dectime_log():
    segmenter_paths.dectime_log.unlink(missing_ok=True)


def get_turn():
    turn = 0

    try:
        turn = len(get_times(segmenter_paths.dectime_log))
    except FileNotFoundError:
        pass
    return turn


def make_decode_cmd(threads=1):
    cmd = (f'bin/ffmpeg -hide_banner -benchmark '
           f'-codec hevc '
           f'{"" if not threads else f"-threads {threads} "}'
           f'-i {segmenter_paths.segment_video.as_posix()} '
           f'-f null -')
    cmd = f'bash -c "{cmd}"'

    return cmd


def iter_decode():
    for ctx.name in ctx.name_list:
        for ctx.projection in ctx.projection_list:
            for ctx.quality in ctx.quality_list:
                for ctx.tiling in ctx.tiling_list:
                    for ctx.tile in ctx.tile_list:
                        for ctx.chunk in ctx.chunk_list:
                            yield
