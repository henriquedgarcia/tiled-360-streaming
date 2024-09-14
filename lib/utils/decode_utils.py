from config.config import config
from lib.assets.context import ctx
from lib.assets.errors import AbortError, DecodeOkError
from lib.assets.logger import logger
from lib.assets.paths import paths
from lib.utils.util import decode_video
from lib.utils.util import get_times, print_error
from .segmenter_utils import assert_one_chunk_video


def iter_decode():
    for ctx.name in ctx.name_list:
        for ctx.projection in ctx.projection_list:
            for ctx.quality in ctx.quality_list:
                for ctx.tiling in ctx.tiling_list:
                    for ctx.tile in ctx.tile_list:
                        for ctx.chunk in ctx.chunk_list:
                            yield


def decode_chunks():
    for ctx.attempt in range(config.decoding_num):
        for _ in iter_decode():
            print(f'==== Decoding {ctx} ====')
            try:
                decode()
            except (DecodeOkError, AbortError) as e:
                print_error(f'\t{e.args[0]}')


def decode():
    print(f'\tAttempt {ctx.attempt + 1}/{config.decoding_num}')
    print(f'\tChecking dectime')
    check_dectime()

    print(f'\tChecking chunks')
    check_chunk()

    print(f'\tDecoding {ctx.turn+1}/{config.decoding_num}')
    decode_decode()


def decode_decode():
    dectime_log = paths.dectime_log
    folder = dectime_log.parent
    chunk_video = paths.chunk_video

    stdout = decode_video(chunk_video, threads=1)

    if not folder.exists():
        folder.mkdir(parents=True)

    with open(dectime_log, 'a') as f:
        f.write('\n' + stdout)

    check_dectime()
    raise AbortError(f'Decoded {ctx.turn} times.')


def check_dectime():
    try:
        assert_dectime()
    except FileNotFoundError:
        logger.update_status('dectime_ok', False)
        return
    finally:
        logger.update_status('decode_turn', ctx.turn)

    if ctx.turn >= config.decoding_num:
        logger.update_status('dectime_ok', True)
        raise DecodeOkError(f'Dectime is OK. Skipping.')


def assert_dectime():
    if not logger.get_status('dectime_ok'):
        assert_dectime_log()


def get_turn():
    turn = len(get_times(paths.dectime_log))
    if turn == 0:
        clean_dectime_log()
    return turn


def assert_dectime_log():
    try:
        ctx.turn = get_turn()
    except FileNotFoundError:
        ctx.turn = 0
        raise FileNotFoundError('dectime_log not exist.')


def check_chunk():
    try:
        assert_chunk()
    except FileNotFoundError:
        print_error(f'\tChunks not Found.')
        logger.register_log('\tChunk not exist.', paths.chunk_video)
        raise AbortError(f'Chunk not exist.')
    logger.update_status('chunk_ok', True)


def assert_chunk():
    if not logger.get_status('chunk_ok'):
        assert_one_chunk_video()


def clean_dectime_log():
    paths.dectime_log.unlink(missing_ok=True)


def make_decode_cmd(threads=1):
    cmd = (f'bin/ffmpeg -hide_banner -benchmark '
           f'-codec hevc '
           f'{"" if not threads else f"-threads {threads} "}'
           f'-i {paths.segment_video.as_posix()} '
           f'-f null -')
    cmd = f'bash -c "{cmd}"'

    return cmd
