from subprocess import run, STDOUT, PIPE

from config.config import config
from lib.utils.segmenter_utils import skip_compress, skip_segmenter, tile_position
from lib.utils.util import splitx, print_error
from .assets.context import ctx
from .assets.logger import logger
from .assets.paths import paths
from .assets.worker import Worker


class Segmenter(Worker):
    def main(self):
        for ctx.name in ctx.name_list:
            for ctx.projection in ctx.projection_list:
                prepare()

                for ctx.quality in ctx.quality_list:
                    for ctx.tiling in ctx.tiling_list:
                        for ctx.tile in ctx.tile_list:
                            compress()
                            segmenter()


def prepare():
    print(f'==== Preparing {ctx} ====')
    if paths.lossless_file.exists():
        print_error(f'\tThe file {paths.lossless_file} exist. Skipping.')
        return

    if not paths.original_file.exists():
        logger.register(f'The original_file not exist.', paths.original_file)
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

    x1, y1, x2, y2 = tile_position()
    qp_options = (':ipratio=1:pbratio=1'
                  if config.rate_control == 'qp' else '')

    cmd = 'bin/ffmpeg -hide_banner -y -psnr '
    cmd += f'-i {paths.lossless_file.as_posix()} '
    cmd += f'-c:v libx265 '
    cmd += f'-{config.rate_control} {ctx.quality} -tune psnr '
    cmd += f'-x265-params '
    cmd += (f'keyint={config.gop}:min-keyint={config.gop}:'
            f'open-gop=0:scenecut=0:info=0{qp_options} ')
    cmd += f'-vf crop=w={x2 - x1}:h={y2 - y1}:x={x1}:y={y1} '
    cmd += f'{paths.compressed_file.as_posix()}'

    cmd = f'bash -c "{cmd}"'

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
    cmd = 'bin/MP4Box '
    cmd += '-split 1 '
    cmd += f'{paths.compressed_file.as_posix()} '
    cmd += f"-out {paths.segments_folder.as_posix()}/tile{ctx.tile}_'$'num%03d$.mp4 "
    # cmd += f'2>&1 {self.segment_log.as_posix()}'

    cmd = f'bash -c "{cmd} &> {paths.segmenter_log.as_posix()}"'

    print(cmd)
    paths.segmenter_log.parent.mkdir(parents=True, exist_ok=True)
    process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    paths.segmenter_log.write_text(process.stdout)
