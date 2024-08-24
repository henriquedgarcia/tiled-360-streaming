import shutil
from subprocess import run, STDOUT, PIPE

from config.config import config
from lib.assets.context import ctx
from lib.assets.logger import logger
from lib.assets.paths import paths
from lib.utils.util import print_error, decode_file, splitx, run_command


def iterate_name_projection_quality_tiling_tile():
    for ctx.name in ctx.name_list:
        for ctx.projection in ctx.projection_list:
            for ctx.quality in ctx.quality_list:
                for ctx.tiling in ctx.tiling_list:
                    for ctx.tile in ctx.tile_list:
                        yield


def __compress__(): ...


def create_compress():
    check_compress()

    if logger.get_status('compressed_ok'):
        return

    print(f'==== Compress {ctx} ====')
    cmd = compress()
    print('\t' + cmd)
    run_command(cmd, paths.compressed_video.parent, paths.compressed_log)
    check_compress()


def compress():
    lossless_file = paths.lossless_file.as_posix()
    compressed_file = paths.compressed_video.as_posix()

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


def check_compress(decode=False):
    if not logger.get_status('compressed_ok'):
        check_compressed()

    if decode and logger.get_status('compressed_ok'):
        check_decode_compressed()

    if logger.get_status('compressed_ok'):
        return True

    if not logger.get_status('lossless_ok'):
        check_lossless_video()

    return not logger.get_status('lossless_ok')


def check_compressed():
    try:
        check_compressed_video()
        check_compressed_log()
        logger.update_status('compressed_ok', True)
    except FileNotFoundError:
        clean_compress()
        logger.update_status('compressed_ok', False)


def check_decode_compressed():
    print_error(f'\tDecoding Compressed Video... ', end='')

    if logger.get_status('compressed_decode_ok'):
        print_error(f'OK')
        return

    try:
        check_decode_compressed_video()
        logger.update_status('compressed_decode_ok', True)
    except FileNotFoundError:
        clean_compress()


def check_compressed_video():
    compressed_file_size = paths.compressed_video.stat().st_size
    if compressed_file_size == 0:
        print_error(f'\tcompressed_file_size == 0.')
        logger.register_log('compressed_file_size == 0', paths.compressed_video)
        raise FileNotFoundError('compressed_file_size == 0')


def check_compressed_log():
    compressed_log_text = paths.compressed_log.read_text()

    if 'encoded 1800 frames' not in compressed_log_text:
        logger.register_log('compressed_log is corrupt', paths.compressed_log)
        print_error(f'\tThe file {paths.compressed_log} is corrupt. Cleaning.')
        raise FileNotFoundError('compressed_log is corrupt')

    if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
        logger.register_log('CODEC ERROR', paths.compressed_log)
        print_error(f'\tThe file {paths.compressed_log} have codec different of Lavc59.18.100 libx265. Skipping.')
        raise FileNotFoundError('CODEC ERROR')


def check_lossless_video():
    logger.update_status('lossless_ok', True)
    if not paths.lossless_file.exists():
        logger.register_log(f'The lossless_file not exist.', paths.lossless_file)
        print_error(f'\tCant create the compressed video. The lossless video not exist. Skipping.')
        logger.update_status('lossless_ok', False)


def check_decode_compressed_video():
    stdout = decode_file(paths.compressed_video)
    if "frame= 1800" not in stdout:
        logger.register_log(f'Decoding Compress Error.',
                            paths.compressed_video)
        print_error(f'\tDecode Compressed Video Error. Cleaning.')
        raise FileNotFoundError('Decoding Compress Error')
    return stdout


def clean_compress():
    paths.compressed_log.unlink(missing_ok=True)
    paths.compressed_video.unlink(missing_ok=True)
    logger.update_status('compressed_decode_ok', False)
    logger.update_status('compressed_ok', False)


def __segment__(): ...


def create_segments():
    for _ in iterate_name_projection_quality_tiling_tile():
        check_segmenter()

        if logger.get_status('segments_ok'):
            continue

        print(f'==== Segment {ctx} ====')
        cmd = segmenter()
        print('\t' + cmd)
        run_command(cmd, paths.chunks_folder, paths.segmenter_log)
        check_segmenter()


def check_segmenter(decode=False):
    if not logger.get_status('segments_ok'):
        check_segment()

    if decode and logger.get_status('segments_ok'):
        check_decode_segments()

    if not logger.get_status('compressed_ok'):
        create_compress()

    return not logger.get_status('compressed_ok')


def check_segment():
    try:
        check_segment_log()
        check_segment_video()
        logger.update_status('segments_ok', True)
    except FileNotFoundError:
        clean_segments()
        logger.update_status('segments_ok', False)


def check_segment_log():
    segment_log_txt = paths.segmenter_log.read_text()
    ctx.chunk = f'{config.n_chunks - 1}'
    segment_video = paths.segment_video.as_posix()
    ctx.chunk = None

    if f'{segment_video}' not in segment_log_txt:
        logger.register_log('Segment_log is corrupt. Cleaning', paths.segmenter_log)
        print_error(f'\tThe file {paths.segmenter_log} is corrupt. Cleaning.')
        raise FileNotFoundError


def check_segment_video():
    for ctx.chunk in ctx.chunk_list:
        segment_file_size = paths.segment_video.stat().st_size

        if segment_file_size == 0:
            logger.register_log(f'The segment_file SIZE == 0', paths.segment_video)
            print_error(f'\tSegmentlog is OK but the file size == 0. Cleaning.')
            ctx.chunk = None
            raise FileNotFoundError
    ctx.chunk = None


def clean_segments():
    paths.segmenter_log.unlink(missing_ok=True)
    shutil.rmtree(paths.chunks_folder, ignore_errors=True)


def check_decode_segments():
    print_error(f'\tDecoding Segment Video... ', end='')

    if logger.get_status('segments_decode_ok'):
        print_error(f'OK')
        return

    for ctx.chunk in ctx.chunk_list:
        try:
            check_decode_segments_video()
        except FileNotFoundError:
            clean_segments()
            logger.update_status('segments_decode_ok', False)
            logger.update_status('segments_ok', False)
            break
    else:
        logger.update_status('segments_decode_ok', True)

    ctx.chunk = None


def check_decode_segments_video():
    stdout = decode_file(paths.segment_video)
    if "frame=   30" not in stdout:  # specific for ffmpeg 5.0
        logger.register_log(f'Segment Decode Error.',
                            paths.segment_video)
        print_error(f'Segment Decode Error. Cleaning..')
        raise FileNotFoundError(f'Decoding Segment Error.')
    return stdout


def segmenter():
    compressed_file = paths.compressed_video.as_posix()
    chunks_folder = paths.chunks_folder.as_posix()
    cmd = ('bash -c '
           '"'
           f'ffmpeg -hide_banner -i {compressed_file} '
           '-c copy -f segment -segment_time 1 -reset_timestamps 1 '
           f'{chunks_folder}/tile{ctx.tile}_%03d.hevc'
           '"')

    return cmd


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


def check_chunk_file(decode=False):
    try:
        segment_file_size = paths.segment_video.stat().st_size
        assert segment_file_size > 0
    except FileNotFoundError:
        print_error(f'The segment not exist. ')
        logger.log("segment_file not exist.", paths.segment_video)
        raise FileNotFoundError
    except AssertionError:
        logger.register_log(f'The segment_file SIZE == 0', paths.segment_video)
        print_error(f'\tSegmentlog is OK but the file size == 0. Cleaning.')
        raise FileNotFoundError

    if decode:
        stdout = decode_file(paths.segment_video)
        if "frame=   30" not in stdout:  # specific for ffmpeg 5.0
            print_error(f'Segment Decode Error. Cleaning..')
            logger.register_log(f'Segment Decode Error.', paths.segment_video)
            raise FileNotFoundError


def prepare():
    """
    deprecated
    :return:
    """
    print(f'==== Preparing {ctx} ====')
    if paths.lossless_file.exists():
        print_error(f'\tThe file {paths.lossless_file} exist. Skipping.')
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
           f'{paths.lossless_file.as_posix()}'
           f'"')

    print('\t', cmd)

    paths.lossless_file.parent.mkdir(parents=True, exist_ok=True)
    process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    paths.lossless_log.write_text(process.stdout)
