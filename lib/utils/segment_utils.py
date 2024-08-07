from lib import config
from lib.assets import paths, logger, ctx
from lib.utils import print_error, decode_file, splitx


def skip_compress(decode=False):
    try:
        check_compressed_video()
        check_compressed_log()
        if decode: check_decode_compressed_video()
        return True  # all ok

    except FileNotFoundError:
        clean_compress()

    try:
        check_lossless_video()
    except FileNotFoundError:
        return True
    return False


def check_compressed_video():
    compressed_file_size = paths.compressed_file.stat().st_size
    if compressed_file_size == 0:
        print_error(f'\tcompressed_file_size == 0.')
        raise FileNotFoundError('compressed_file_size == 0')


def check_compressed_log():
    compressed_log_text = paths.compressed_log.read_text()

    if 'encoded 1800 frames' not in compressed_log_text:
        logger.register('compressed_log is corrupt', paths.compressed_log)
        print_error(f'\tThe file {paths.compressed_log} is corrupt. Skipping.')
        raise FileNotFoundError('compressed_log is corrupt')

    if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
        logger.register('CODEC ERROR', paths.compressed_log)
        print_error(f'\tThe file {paths.compressed_log} have codec different of Lavc59.18.100 libx265. Skipping.')
        raise FileNotFoundError('CODEC ERROR')


def check_lossless_video():
    if not paths.lossless_file.exists():
        logger.register(f'The lossless_file not exist.', paths.lossless_file)
        print_error(f'\tCant create the compressed video. The lossless video not exist. Skipping.')
        raise FileNotFoundError('Decoding Compress Error')


def check_decode_compressed_video():
    stdout = decode_file(paths.compressed_file)
    if "frame= 1800" not in stdout:
        logger.register(f'Decoding Compress Error.', paths.compressed_file)
        print_error(f'\tDecode Compressed Video Error. Cleaning.')
        raise FileNotFoundError('Decoding Compress Error')


def clean_compress():
    paths.compressed_log.unlink(missing_ok=True)
    paths.compressed_file.unlink(missing_ok=True)


def skip_segmenter(decode=False):
    try:
        check_segment_log()
        check_segment_video()
        if decode: check_decode_compressed_video()
        return True  # all ok

    except FileNotFoundError:
        clean_segments()

    try:
        check_compressed_video()
    except FileNotFoundError:
        return True

    return False


def check_segment_log():
    segment_log_txt = paths.segmenter_log.read_text()
    if 'file 60 done' not in segment_log_txt:
        logger.register('Segment_log is corrupt. Cleaning', paths.segmenter_log)
        print_error(f'\tThe file {paths.segmenter_log} is corrupt. Cleaning.')
        raise FileNotFoundError


def check_segment_video(decode=False):
    for ctx.chunk in ctx.chunk_list:
        segment_file_size = paths.segment_file.stat().st_size

        if segment_file_size == 0:
            logger.register(f'The segment_file SIZE == 0', paths.segment_file)
            print_error(f'\tSegmentlog is OK but the file size == 0. Cleaning.')
            raise FileNotFoundError

        # decodifique os segmentos
        if decode:
            stdout = decode_file(paths.segment_file)

            if "frame=   30" not in stdout:  # specific for ffmpeg 5.0
                print_error(f'Segment Decode Error. Cleaning..')
                logger.register(f'Segment Decode Error.',
                                paths.segment_file)
                raise FileNotFoundError


def clean_segments():
    paths.segmenter_log.unlink(missing_ok=True)
    for ctx.chunk in ctx.chunk_list:
        paths.segment_file.unlink(missing_ok=True)


def tile_position():
    """
    Need video, tiling and tile
    :return: x1, x2, y1, y2
    """
    proj_h, proj_w = config.video_shape
    tiling_w, tiling_h = splitx(ctx.tiling)
    tile_w, tile_h = int(proj_w / tiling_w), int(proj_h / tiling_h)
    tile_m, tile_n = int(ctx.tile) % tiling_w, int(ctx.tile) // tiling_w
    x1 = tile_m * tile_w
    y1 = tile_n * tile_h
    x2 = tile_m * tile_w + tile_w  # not inclusive [...)
    y2 = tile_n * tile_h + tile_h  # not inclusive [...)
    return x1, y1, x2, y2


def check_chunk_file(decode=False):
    try:
        segment_file_size = paths.segment_file.stat().st_size
        assert segment_file_size > 0
    except FileNotFoundError:
        print_error(f'The segment not exist. ')
        logger.log("segment_file not exist.", paths.segment_file)
        raise FileNotFoundError
    except AssertionError:
        logger.register(f'The segment_file SIZE == 0', paths.segment_file)
        print_error(f'\tSegmentlog is OK but the file size == 0. Cleaning.')
        raise FileNotFoundError

    if decode:
        stdout = decode_file(paths.segment_file)
        if "frame=   30" not in stdout:  # specific for ffmpeg 5.0
            print_error(f'Segment Decode Error. Cleaning..')
            logger.register(f'Segment Decode Error.', paths.segment_file)
            raise FileNotFoundError
