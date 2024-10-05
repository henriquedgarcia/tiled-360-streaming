from contextlib import contextmanager


@contextmanager
def context_chunk(ctx, value):
    c = ctx.chunk
    ctx.chunk = f'{value}'
    try:
        yield
    finally:
        ctx.chunk = c


@contextmanager
def context_tile(ctx, value):
    t = ctx.tile
    ctx.tile = f'{value}'
    try:
        yield
    finally:
        ctx.tile = t


@contextmanager
def context_quality(ctx, quality='0', rate_control='crf'):
    qlt = ctx.quality
    rc = ctx.config.rate_control

    ctx.quality = quality
    ctx.config.rate_control = rate_control

    try:
        yield
    finally:
        ctx.quality = qlt
        ctx.config.rate_control = rc
