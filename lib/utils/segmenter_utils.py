import shutil
from subprocess import run, STDOUT, PIPE

from config.config import config
from lib.assets.context import ctx
from lib.assets.errors import AbortError
from lib.assets.logger import logger
from lib.assets.paths import paths
from lib.utils.context_utils import context_chunk
from lib.utils.util import print_error, decode_video, splitx, run_command


def iterate_name_projection_quality_tiling_tile():
    for ctx.name in ctx.name_list:
        for ctx.projection in ctx.projection_list:
            for ctx.quality in ctx.quality_list:
                for ctx.tiling in ctx.tiling_list:
                    for ctx.tile in ctx.tile_list:
                        yield


def create_segments(decode_check=False):
    for _ in iterate_name_projection_quality_tiling_tile():
        print(f'==== Segmenter {ctx} ====')
        try:
            segmenter(decode_check=decode_check)
        except AbortError as e:
            print_error(f'\t{e.args[0]}')


def segmenter(decode_check=False):
    print(f'\tChecking chunks')
    check_chunk(decode_check=decode_check)

    print(f'\tChecking tiles')
    try:
        check_tile(decode_check=decode_check)
    except FileNotFoundError:
        print(f'\tCreating Tiles ')
        make_tiles(decode_check=decode_check)

    print(f'\tChunking Tiles ')
    make_chunks(decode_check=decode_check)


def __Make_tiles__(): ...


def check_tile(decode_check=False):
    try:
        assert_tiles(decode_check=decode_check)
    except FileNotFoundError:
        print(f'\tTiles not Found.')
        print(f'\tChecking lossless')
        assert_lossless()
        raise FileNotFoundError


def make_tiles(decode_check=False):
    cmd = make_compress_tile_cmd()
    print('\t' + cmd)
    run_command(cmd, paths.tiles_folder, paths.tile_log)

    try:
        assert_tiles()
    except FileNotFoundError:
        raise AbortError(f'Cant create Compressed video.')


def assert_tiles(decode_check=False):
    print(f'\tChecking tiles.')
    if not logger.get_status('tile_ok'):
        try:
            assert_tile_log()
            assert_tile_video()
            logger.update_status('tile_ok', True)

            if decode_check:
                assert_tile_decode()
        except FileNotFoundError as e:
            clean_tile()
            logger.update_status('tile_ok', False)
            logger.update_status('tile_decode_ok', False)
            raise e

    return 'all ok'


def assert_tile_log():
    try:
        compressed_log_text = paths.tile_log.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f'Tile log not exist.')

    if 'encoded 1800 frames' not in compressed_log_text:
        logger.register_log('Tile log is corrupt', paths.tile_log)
        raise FileNotFoundError('Tile log is corrupt')

    if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
        logger.register_log('Codec version is different.', paths.tile_log)
        raise FileNotFoundError('Codec version is different.')

    return 'all ok'


def assert_tile_video():
    try:
        compressed_file_size = paths.tile_video.stat().st_size
    except FileNotFoundError:
        raise FileNotFoundError(f'Tile not exist.')

    if compressed_file_size == 0:
        logger.register_log('Tile size == 0.', paths.tile_video)
        raise FileNotFoundError('Tile size == 0.')
    return 'all ok'


def assert_tile_decode():
    print(f'\tDecoding tile', end='')

    if logger.get_status('tile_decode_ok'):
        print_error(f'. OK')
        return 'decode all ok'

    assert_one_tile_decode()
    print_error(f'. OK')
    return 'decode all ok'


def assert_one_tile_decode():
    stdout = decode_video(paths.tile_video)
    if "frame= 1800" not in stdout:
        logger.register_log(f'Decoding Compress Error.',
                            paths.tile_video)
        print_error(f'\tDecode Compressed Video Error. Cleaning.')
        raise FileNotFoundError('Decoding Compress Error')
    return stdout


def assert_lossless():
    if not logger.get_status('lossless_ok'):
        if not paths.lossless_video.exists():
            raise AbortError(f'Need create tile but lossless_video not found.')
        logger.update_status('lossless_ok', True)


def make_compress_tile_cmd():
    lossless_file = paths.lossless_video.as_posix()
    compressed_file = paths.tile_video.as_posix()

    x1, y1, x2, y2 = tile_position()

    gop_options = f'keyint={config.gop}:min-keyint={config.gop}:open-gop=0'
    misc_options = f'scenecut=0:info=0'
    qp_options = ':ipratio=1:pbratio=1' if config.rate_control == 'qp' else ''
    lossless_option = ':lossless=1' if ctx.quality == '0' else ''

    cmd = ('bash -c '
           '"'
           'bin/ffmpeg -hide_banner -y -psnr '
           f'-i {lossless_file} '
           f'-{config.rate_control} {ctx.quality} -tune psnr '
           f'-c:v libx265 '
           f'-x265-params {gop_options}:{misc_options}{qp_options}{lossless_option} '
           f'-vf crop=w={x2 - x1}:h={y2 - y1}:x={x1}:y={y1} '
           f'{compressed_file}'
           f'"')

    return cmd


def clean_tile():
    paths.tile_log.unlink(missing_ok=True)
    paths.tile_video.unlink(missing_ok=True)


def __assert_chunks__(): ...


def check_chunk(decode_check=False):
    try:
        assert_chunks(decode_check=decode_check)
        raise AbortError('Chunks are OK.')
    except FileNotFoundError:
        print(f'\tChunks not Found.')
        pass


def make_chunks(decode_check=False):
    cmd = make_segmenter_cmd()
    print('\t' + cmd)
    run_command(cmd, paths.chunks_folder, paths.segmenter_log)

    try:
        assert_chunks()
    except FileNotFoundError:
        raise AbortError('Error creating chunks. See log.')


def assert_chunks(decode_check=False):
    if not logger.get_status('segmenter_ok'):
        try:
            assert_segmenter_log()
            assert_chunks_video()
            logger.update_status('segmenter_ok', True)

            if decode_check:
                assert_chunks_decode()
                logger.update_status('chunks_decode_ok', True)

        except FileNotFoundError as e:
            clean_segmenter()
            logger.update_status('segmenter_ok', False)
            logger.update_status('chunks_decode_ok', False)
            raise e

    return 'all ok.'


def assert_segmenter_log():
    try:
        segment_log_txt = paths.segmenter_log.read_text()
    except FileNotFoundError:
        raise FileNotFoundError('Segmenter log not exist.')

    with context_chunk(f'{config.n_chunks - 1}'):
        segment_video = paths.chunk_video.as_posix()
    # gambiarra. Todos os logs do teste estão com as pastas antigas.
    segment_video_changed = f'{segment_video}'.replace('chunks', 'segments')

    if f'{segment_video}' not in segment_log_txt and f'{segment_video_changed}' not in segment_log_txt:
        logger.register_log('Segmenter log is corrupt.', paths.segmenter_log)
        raise FileNotFoundError('Segmenter log is corrupt.')

    return 'all ok'


def assert_chunks_video():
    with context_chunk(None):
        for ctx.chunk in ctx.chunk_list:
            segment_video = paths.chunk_video

            try:
                segment_file_size = segment_video.stat().st_size
            except FileNotFoundError:
                raise FileNotFoundError(f'video chunk{ctx.chunk} not exist.')

            if segment_file_size == 0:
                logger.register_log(f'Chunk video size == 0', segment_video)
                raise FileNotFoundError('Chunk video size == 0.')

    return 'all ok'


def assert_chunks_decode():
    print(f'\tDecoding chunks', end='')

    if logger.get_status('chunks_decode_ok'):
        print(f'. OK')
        return 'decode all ok'

    with context_chunk(None):
        for ctx.chunk in ctx.chunk_list:
            print('.', end='')
            assert_one_chunk_decode()

    print(f'. OK')
    return 'decode all ok'


def assert_one_chunk_decode():
    stdout = decode_video(paths.chunk_video)
    if "frame=   30" not in stdout:  # specific for ffmpeg 5.0
        logger.register_log(f'Segment Decode Error.',
                            paths.chunk_video)
        raise FileNotFoundError(f'Chunk Decode Error.')
    return stdout


def make_segmenter_cmd():
    compressed_file = paths.tile_video.as_posix()
    chunks_folder = paths.chunks_folder.as_posix()
    cmd = ('bash -c '
           '"'
           f'ffmpeg -hide_banner -i {compressed_file} '
           '-c copy -f segment -segment_time 1 -reset_timestamps 1 '
           f'{chunks_folder}/tile{ctx.tile}_%03d.hevc'
           '"')

    return cmd


def clean_segmenter():
    paths.segmenter_log.unlink(missing_ok=True)
    shutil.rmtree(paths.chunks_folder, ignore_errors=True)


def __others__(): ...


def tile_position():
    """
    Need video, tiling and tile
    :return: x1, x2, y1, y2
    """
    proj_h, proj_w = ctx.video_shape
    tiling_w, tiling_h = splitx(ctx.tiling)
    tile_w, tile_h = int(proj_w / tiling_w), int(proj_h / tiling_h)
    tile_m, tile_n = int(ctx.tile) % tiling_w, int(ctx.tile) // tiling_w
    x1 = tile_m * tile_w
    y1 = tile_n * tile_h
    x2 = tile_m * tile_w + tile_w  # not inclusive [...)
    y2 = tile_n * tile_h + tile_h  # not inclusive [...)
    return x1, y1, x2, y2


def prepare():
    """
    deprecated
    :return:
    """
    print(f'==== Preparing {ctx} ====')
    if paths.lossless_video.exists():
        print_error(f'\tThe file {paths.lossless_video} exist. Skipping.')
        return

    if not paths.original_file.exists():
        logger.register_log(f'The original_file not exist.', paths.original_file)
        print_error(f'\tThe file {paths.original_file=} not exist. Skipping.')
        return

    resolution_ = splitx(ctx.scale)
    dar = resolution_[0] / resolution_[1]

    cmd = (f'bash -c '
           f'"'
           f'bin/ffmpeg '
           f'-hide_banner -y '
           f'-ss {ctx.offset} '
           f'-i {config.original_file.as_posix()} '
           f'-crf 0 '
           f'-t {config.duration} '
           f'-r {config.fps} '
           f'-map 0:v '
           f'-vf scale={ctx.scale},setdar={dar} '
           f'{paths.lossless_video.as_posix()}'
           f'"')

    print('\t', cmd)

    paths.lossless_video.parent.mkdir(parents=True, exist_ok=True)
    process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    paths.lossless_log.write_text(process.stdout)
