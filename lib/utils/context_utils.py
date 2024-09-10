from contextlib import contextmanager

from lib.assets.context import ctx


@contextmanager
def context_chunk(value):
    c = ctx.chunk
    ctx.chunk = f'{value}'
    try:
        yield
    finally:
        ctx.chunk = c
