from contextlib import contextmanager
from time import time

from lib.assets.errors import AbortError
from lib.utils.util import print_error


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


@contextmanager
def task(self, verbose=True):
    if verbose:
        print(f'\n==== {self.__class__.__name__} {self.ctx} ====')
    try:
        yield
    except AbortError as e:
        msg = e.args[0]
        if msg:
            print_error(f'\t{e.args[0]}')
    finally:
        pass


@contextmanager
def timer(ident=0):
    start = time()
    ident = '\t' * ident

    try:
        yield
    finally:
        print(f"{ident}time={time() - start}.")
