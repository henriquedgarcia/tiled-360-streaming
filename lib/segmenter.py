from subprocess import run, STDOUT, PIPE

from config.config import config
from lib.utils.segmenter_utils import skip_compress, skip_segmenter, tile_position
from lib.utils.util import splitx, print_error
from .assets.context import ctx
from .assets.logger import logger
from .assets.paths import paths
from .assets.worker import Worker


def create_compress():
    for ctx.name in ctx.name_list:
        for ctx.projection in ctx.projection_list:
            for ctx.quality in ctx.quality_list:
                for ctx.tiling in ctx.tiling_list:
                    for ctx.tile in ctx.tile_list:
                        compress()
                        # segmenter()


def create_segments():
    for ctx.name in ctx.name_list:
        for ctx.projection in ctx.projection_list:
            for ctx.quality in ctx.quality_list:
                for ctx.tiling in ctx.tiling_list:
                    for ctx.tile in ctx.tile_list:
                        segmenter()


class Segmenter(Worker):
    def main(self):
        ctx.quality_list = ['0'] + ctx.quality_list
        create_compress()
        create_segments()


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


def compress():
    print(f'==== Compress {ctx} ====')

    if skip_compress():
        return

    lossless_file = paths.lossless_file.as_posix()
    compressed_file = paths.compressed_file.as_posix()

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

    print(cmd)
    paths.compressed_file.parent.mkdir(parents=True, exist_ok=True)
    process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    paths.compressed_log.write_text(process.stdout)


def segmenter():
    print(f'==== Segmenting {ctx} ====')
    if skip_segmenter(): return

    # todo: Alternative:
    # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_t
    # ime 1 -reset_timestamps 1 output%03d.mp4
    cmd = ('bash -c '
           '"'
           'bin/MP4Box '
           '-split 1 '
           f'{paths.compressed_file.as_posix()} '
           f"-out {paths.segments_folder.as_posix()}/tile{ctx.tile}_'$'num%03d$.mp4 "
           # f'&> {paths.segmenter_log.as_posix()}'
           f'"')
    # cmd += f'2>&1 {self.segment_log.as_posix()}'

    # cmd = f'bash -c "{cmd} &> {paths.segmenter_log.as_posix()}"'

    print(cmd)
    paths.segmenter_log.parent.mkdir(parents=True, exist_ok=True)
    process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    paths.segmenter_log.write_text(process.stdout)
