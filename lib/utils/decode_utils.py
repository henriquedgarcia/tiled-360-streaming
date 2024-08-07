from lib import config
from lib.assets import paths, ctx
from lib.utils import get_times
from lib.utils.segment_utils import check_chunk_file


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
