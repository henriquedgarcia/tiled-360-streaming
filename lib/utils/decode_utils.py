from .segmenter_utils import assert_chunks
from config.config import config
from lib.assets.context import ctx
from lib.assets.errors import AbortError
from lib.assets.logger import logger
from lib.assets.paths import paths
from lib.utils.util import get_times, run_command, print_error


def iter_decode():
    for ctx.name in ctx.name_list:
        for ctx.projection in ctx.projection_list:
            for ctx.quality in ctx.quality_list:
                for ctx.tiling in ctx.tiling_list:
                    for ctx.tile in ctx.tile_list:
                        for ctx.chunk in ctx.chunk_list:
                            yield


def decode_chunks():
    for ctx.turn in range(config.decoding_num):
        for _ in iter_decode():
            print(f'==== Decoding {ctx} ====')
            try:
                decode()
            except AbortError as e:
                print_error(f'\t{e.args[0]}')


def decode():
    print(f'\tChecking dectime')
    get_decode_status()

    print(f'\tChecking chunks')
    check_chunks()

    print(f'\tTurn {ctx.turn}/{config.decoding_num}')

    cmd = make_decode_cmd()
    run_command(cmd, paths.dectime_log.parent, paths.dectime_log, mode='a')

    get_decode_status()
    if not logger.get_status('dectime_ok'):
        raise AbortError(f'Decode {ctx.turn} times.')


def get_decode_status():
    try:
        assert_dectime()
        logger.update_status('dectime_ok', True)
        raise AbortError(f'\tDectime is OK. Skipping.')
    except FileNotFoundError:
        print_error(f'\tDectime not Found.')
        logger.update_status('dectime_ok', False)


def assert_dectime():
    if logger.get_status('decode_ok'):
        raise AbortError(f'\tDectime is OK. Skipping.')



def check_chunks():
    try:
        assert_chunks()
        logger.update_status('segmenter_ok', True)
    except FileNotFoundError:
        print_error(f'\tChunks not Found.')
        logger.update_status('segmenter_ok', False)
        logger.register_log('\tChunk not exist.', paths.chunk_video)
        raise AbortError(f'Chunk not exist.')


def assert_dectime_log():
    ctx.turn = get_turn()
    logger.update_status('decode_turn', ctx.turn)

    if ctx.turn == 0:
        clean_dectime_log()

    if logger.get_status('decode_turn') > config.decoding_num:
        logger.update_status('decode_ok', True)


def clean_dectime_log():
    paths.dectime_log.unlink(missing_ok=True)


def get_turn():
    turn = 0

    try:
        turn = len(get_times(paths.dectime_log))
    except FileNotFoundError:
        pass
    return turn


def make_decode_cmd(threads=1):
    cmd = (f'bin/ffmpeg -hide_banner -benchmark '
           f'-codec hevc '
           f'{"" if not threads else f"-threads {threads} "}'
           f'-i {paths.segment_video.as_posix()} '
           f'-f null -')
    cmd = f'bash -c "{cmd}"'

    return cmd
