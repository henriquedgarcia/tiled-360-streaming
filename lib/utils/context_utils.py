from contextlib import contextmanager

from config.config import config
from lib.assets.context import ctx


@contextmanager
def context_chunk(value):
    c = ctx.chunk
    ctx.chunk = f'{value}'
    try:
        yield
    finally:
        ctx.chunk = c


@contextmanager
def context_quality(quality='0', rate_control='crf'):
    qlt = ctx.quality
    rc = config.rate_control

    ctx.quality = quality
    config.rate_control = rate_control

    try:
        yield
    finally:
        ctx.quality = qlt
        ctx.rate_control = rc
