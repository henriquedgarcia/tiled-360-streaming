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
def context_quality(ctx, config, quality='0', rate_control='crf'):
    qlt = ctx.quality
    rc = config.rate_control

    ctx.quality = quality
    config.rate_control = rate_control

    try:
        yield
    finally:
        ctx.quality = qlt
        ctx.rate_control = rc
