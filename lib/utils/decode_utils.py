from config.config import config
from lib.assets.context import ctx
from lib.assets.paths import paths
from lib.utils.segmenter_utils import check_chunk_file
from lib.utils.util import get_times


def skip_decode():
    try:
        check_dectime_log()
    except FileNotFoundError:
        try:
            check_chunk_file()
        except FileNotFoundError:
            return True
        return False
    return True  # all ok


def clean_dectime_log():
    paths.dectime_log.unlink(missing_ok=True)


def check_dectime_log():
    dectime_log_content = paths.dectime_log.read_text(encoding='utf-8')
    ctx.turn = len(get_times(dectime_log_content))
    if ctx.turn < config.decoding_num:
        raise FileNotFoundError
    print(f'\tDecoded {ctx.turn}.')
