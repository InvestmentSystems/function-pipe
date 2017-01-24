

import sys
import argparse
from functools import reduce

import numpy as np
import function_pipe as fpn


class PixelFontInput(fpn.PipeNodeInput):

    SHAPE = (5,5)

    def __init__(self, pixel='*', scale=1):
        super().__init__()
        self.scale = scale
        self.pixel = pixel


@fpn.pipe_node
def frame(**kwargs):
    pfi = kwargs[fpn.PN_INPUT]
    shape = tuple(s * pfi.scale for s in pfi.SHAPE)
    return np.zeros(shape=shape, dtype=bool)


@fpn.pipe_node
def v_line(**kwargs):
    pfi = kwargs[fpn.PN_INPUT]
    m = kwargs[fpn.PREDECESSOR_RETURN].copy()
    m[:, slice(0, pfi.scale)] = True
    return m

@fpn.pipe_node
def h_line(**kwargs):
    pfi = kwargs[fpn.PN_INPUT]
    m = kwargs[fpn.PREDECESSOR_RETURN].copy()
    m[slice(0, pfi.scale), :] = True
    return m


@fpn.pipe_node_factory
def v_shift(steps, **kwargs):
    if kwargs[fpn.PREDECESSOR_PN].unwrap == v_line.unwrap:
        raise Exception('cannot v_shift a v_line')
    pfi = kwargs[fpn.PN_INPUT]
    return np.roll(kwargs[fpn.PREDECESSOR_RETURN], pfi.scale * steps, axis=0)

@fpn.pipe_node_factory
def h_shift(steps, **kwargs):
    if kwargs[fpn.PREDECESSOR_PN].unwrap == h_line.unwrap:
        raise Exception('cannot h_shift an h_line')
    pfi = kwargs[fpn.PN_INPUT]
    return np.roll(kwargs[fpn.PREDECESSOR_RETURN], pfi.scale * steps, axis=1)



@fpn.pipe_node_factory
def flip(*coords, **kwargs):
    pfi = kwargs[fpn.PN_INPUT]
    m = kwargs[fpn.PREDECESSOR_RETURN].copy()
    for coord in coords: # x, y pairs
        start = [i * pfi.scale for i in coord]
        end = [i + pfi.scale for i in start]
        iloc = slice(start[1], end[1]), slice(start[0], end[0])
        m[iloc] = ~m[iloc]
    return m


@fpn.pipe_node_factory
def union(*args, **kwargs):
    return reduce(np.logical_or, args)

@fpn.pipe_node_factory
def intersect(*args, **kwargs):
    return reduce(np.logical_and, args)

@fpn.pipe_node_factory
def concat(*args, **kwargs):
    pfi = kwargs[fpn.PN_INPUT]
    space = np.zeros(shape=(pfi.SHAPE[0] * pfi.scale, 1 * pfi.scale),
            dtype=bool)
    concat = lambda x, y: np.concatenate((x, space, y), axis=1)
    return reduce(concat, args)



@fpn.pipe_node
def display(**kwargs):
    pfi = kwargs[fpn.PN_INPUT]
    m = kwargs[fpn.PREDECESSOR_RETURN]
    for row in m:
        for pixel in row:
            if pixel:
                print(pfi.pixel, end='')
            else:
                print(' ', end='')
        print()
    return m

# font based on http://www.dafont.com/visitor.font
chars = {
    '_' : frame,

    '.' : frame | flip((2,4)),

    'p' : union(
        frame | v_line,
        frame | h_line,
        frame | h_line | v_shift(2),
        ) | flip((4,0), (4,1)),

    'y' : (frame | h_line | v_shift(2) |
            flip((0,0), (0,1), (2,3), (2,4), (4,0), (4,1))),

    '0' : union(
        frame | v_line,
        frame | v_line | h_shift(-1),
        frame | h_line,
        frame | h_line | v_shift(-1),
        ),

    '1' : frame | v_line | h_shift(2) | flip((1,0)),

    '2' : union(
        frame | h_line,
        frame | h_line | v_shift(2),
        frame | h_line | v_shift(4),
        ) | flip((4, 1), (0, 3)),

    '3' : union(
        frame | h_line,
        frame | h_line | v_shift(-1),
        frame | v_line | h_shift(4),
        ) | flip((2, 2), (3, 2)),

    '4' : union(
        frame | h_line | v_shift(2),
        frame | v_line | h_shift(-1),
        ) | flip((0, 0), (0, 1)),

    '5' : union(
        frame | h_line,
        frame | h_line | v_shift(2),
        frame | h_line | v_shift(-1),
        ) | flip((0, 1), (4, 3)),

    '6' : union(
        frame | h_line,
        frame | h_line | v_shift(2),
        frame | h_line | v_shift(-1),
        frame | v_line,
        ) | flip((4, 3)),

    #---------------------------------------------------------------------------

    'a' : union(
            frame | v_line,
            frame | v_line | h_shift(-1),
            frame | h_line,
            frame | h_line | v_shift(2),
            ),

    'b' : union(
            frame | v_line,
            frame | v_line | h_shift(-1),
            frame | h_line,
            frame | h_line | v_shift(-1),
            frame | h_line | v_shift(2),
            ) | flip((4,0), (4,4)),

    'h' : union(
            frame | v_line,
            frame | v_line | h_shift(-1),
            frame | h_line | v_shift(-3),
            ),

    'i' : union(
            frame | h_line,
            frame | h_line | v_shift(-1),
            frame | v_line | h_shift(2),
            ),

    'o' : union(
            frame | v_line,
            frame | v_line | h_shift(-1),
            frame | h_line,
            frame | h_line | v_shift(-1),
            ),

    }


def msg_display_pipeline(msg):
    get_char = lambda char: chars.get(char.lower(), chars['_'])
    return concat(*tuple(map(get_char, msg))) | display


def version_banner(args):

    p = argparse.ArgumentParser(
            description='Display the Python version in a banner',
            )
    p.add_argument('--pixel', default='*',
            help=('Set the character used for each pixel of the banner.')
            )
    p.add_argument('--scale', default=1, type=int,
            help=('Set the pixel scale for the banner.')
            )
    ns = p.parse_args(args)
    assert len(ns.pixel) == 1
    assert ns.scale > 0

    # get pipeline function
    msg = 'py%s.%s.%s' % sys.version_info[:3]
    f = msg_display_pipeline(msg)

    pfi = PixelFontInput(pixel=ns.pixel, scale=ns.scale)
    f[pfi]


if __name__ == '__main__':
    version_banner(sys.argv[1:])







