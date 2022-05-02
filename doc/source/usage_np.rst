Numpy Array Processing with PipeNode
************************************

Introduction
============

This example will present a complete command-line program to print an equal-space, bitmap / pixel-font display of the current Python version (or, with extension, something else more useful). The display will be configurable with (1) a scaling factor and (2) a variable character to be used per pixel. For example:

.. code-block:: none

    % python3 pyv.py --scale=1 --pixel=*

    ****  *   * *****       *****        **   *****
    *   * *   *     *       *   *         *       *
    ***** *****   ***       *****         *   *****
    *       *       *       *   *         *   *
    *       *   *****   *   *****   *     *   *****



    % python3 pyv.py --scale=2 --pixel=.

    ........    ..      ..  ..........              ..........                ....      ..........
    ........    ..      ..  ..........              ..........                ....      ..........
    ..      ..  ..      ..          ..              ..      ..                  ..              ..
    ..      ..  ..      ..          ..              ..      ..                  ..              ..
    ..........  ..........      ......              ..........                  ..      ..........
    ..........  ..........      ......              ..........                  ..      ..........
    ..              ..              ..              ..      ..                  ..      ..
    ..              ..              ..              ..      ..                  ..      ..
    ..              ..      ..........      ..      ..........      ..          ..      ..........
    ..              ..      ..........      ..      ..........      ..          ..      ..........

Tutorial
========

Rather than explicitly defining each character as a fixed bit map, we can use simple ``PipeNode`` functions to define characters as pipeline operations on Boolean NumPy arrays. Operations include creating an empty frame, drawing horizontal or vertical lines, shifting those lines, selectively inverting specific pixels, and taking the union or intersection of any number of frames. Since we want to model linear pipelining of frames through transformational nodes, but also need to expose a ``scale`` parameter to numerous nodes, we will use ``PipeNode`` functions and a ``PipeNodeInput`` instance rather than simple function composition.

We will use the follow imports throughout these examples. The ``numpy`` third-party package can be installed with ``pip``.

.. code-block:: python
    :class: copy-button

    import argparse
    import functools
    import sys

    import numpy as np
    import function_pipe as fpn

In order to minimize the number of ``function_pipe`` stdout logs, we will partial in a forwarding lambda that does not print.

.. code-block:: python
    :class: copy-button

    fpn.pipe_node = functools.partial(
        fpn.pipe_node,
        core_decorator=lambda f: f,
    )
    fpn.pipe_node_factory = functools.partial(
        fpn.pipe_node_factory,
        core_decorator=lambda f: f,
    )


A derived ``PipeNodeInput`` class can specify fixed (as class attributes) or configurable (as arguments passed at initialization and set to instance attributes) parameters, available to all ``PipeNode`` functions when called. For this example, we set a fixed frame shape of 5 by 5 pixels as ``SHAPE``, and expose ``scale`` and ``pixel`` as instance attributes.


.. code-block:: python
    :class: copy-button

    class PixelFontInput(fpn.PipeNodeInput):

        SHAPE = (5,5)

        def __init__(self, pixel="*", scale=1):
            super().__init__()
            self.scale = scale
            self.pixel = pixel


Next, we define ``pipe_node`` decorated functions (that that take no *expression-level arguments*) for creating an empty matrix, a vertical line, and a horizontal line. The ``frame`` function serves in the *innermost* position to provide an empty two-dimensional NumPy array filled with False. In the *innermost* position it only has access to the ``fpn.PN_INPUT`` key-word argument. From the ``fpn.PN_INPUT`` it can read the ``SHAPE`` and ``scale`` attributes to correctly construct the frame. The ``v_line`` and ``h_line`` functions expect a frame passed via ``fpn.PREDECESSOR_RETURN``, and use the ``scale`` attribute from  ``fpn.PN_INPUT`` to write correctly sized Boolean True values in a vertical or horizontal line through the origin (index 0, 0, or the upper left corner) on that frame.

.. code-block:: python
    :class: copy-button

    @fpn.pipe_node(fpn.PN_INPUT)
    def frame(pixel_font_input):
        shape = tuple(v * pixel_font_input.scale for v in pixel_font_input.SHAPE)
        return np.zeros(shape=shape, dtype=bool)

    @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def v_line(pixel_font_input, matrix):
        matrix = matrix.copy()
        matrix[:, slice(0, pixel_font_input.scale)] = True
        return matrix

    @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def h_line(pixel_font_input, matrix):
        matrix = matrix.copy()
        matrix[slice(0, pixel_font_input.scale), :] = True
        return matrix


Next, we can create some transformation functions that, given a frame via ``fpn.PREDECESSOR_PN``, transform and return a new frame. The ``pipe_node_factory`` decorated functions ``v_shift`` and ``h_shift`` use the NumPy roll function to shift the two-dimensional array vertically or horizontally by the ``steps`` argument, passed via *expression-level arguments*. The ``steps`` passed are interpreted at the unit level, and are thus multipled by ``scale`` via ``fpn.PN_INPUT``. As a convenience to users (and catching an error made developing these tools), we check and raise an Exception if we try to do a meaningless shift, such as vertically shifting a vertical line, or horizontall shifting a horizontal line. The ``PipeNode.unwrap`` attribute exposes the *core callable* wrapped by the ``PipeNode``, permitting direct comparison regardless of ``PipeNode`` state.



.. code-block:: python
    :class: copy-button

    @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN, fpn.PREDECESSOR_PN)
    def v_shift(pixel_font_input, matrix, predecessor, steps):
        if predecessor.unwrap == v_line.unwrap:
            raise Exception("cannot v_shift a v_line")
        return np.roll(matrix, pixel_font_input.scale * steps, axis=0)

    @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN, fpn.PREDECESSOR_PN)
    def h_shift(pixel_font_input, matrix, predecessor, steps):
        if predecessor.unwrap == h_line.unwrap:
            raise Exception("cannot h_shift an h_line")
        return np.roll(matrix, pixel_font_input.scale * steps, axis=1)



We will need at times to draw points directly, either setting a False pixel to True or vice versa. The ``pipe_node_factory`` decorated function ``flip`` will, given coordinate pairs in positional arguments, invert the Boolean value found. Again, we use the ``fpn.PN_INPUT`` to get the ``scale`` argument so coordinates can be passed at the unit level, independent of the scale.


.. code-block:: python
    :class: copy-button

    @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def flip(pixel_font_input, matrix, *coords):
        matrix = matrix.copy()
        for coord in coords: # x, y pairs
            start = [i * pixel_font_input.scale for i in coord]
            end = [i + pixel_font_input.scale for i in start]
            iloc = slice(start[1], end[1]), slice(start[0], end[0])
            matrix[iloc] = ~matrix[iloc]
        return matrix


The following ``pipe_node_factory`` decorated functions combine variable numbers of ``PipeNode`` instances passed via positional arguments. The `` union`` and ``intersect`` functions perform logical OR and logical AND, respectively, on all positional arguments. The ``concat`` function concatenates frames into a longer frame, inserting a unit-width space bewteen frames.


.. code-block:: python
    :class: copy-button

    @fpn.pipe_node_factory()
    def union(*args):
        return functools.reduce(np.logical_or, args)

    @fpn.pipe_node_factory()
    def intersect(*args):
        return functools.reduce(np.logical_and, args)

    @fpn.pipe_node_factory(fpn.PN_INPUT)
    def concat(pixel_font_input, *args):
        space = np.zeros(
            shape=(
                pixel_font_input.SHAPE[0] * pixel_font_input.scale,
                1 * pixel_font_input.scale
            ),
            dtype=bool,
        )
        concat = lambda x, y: np.concatenate((x, space, y), axis=1)
        return functools.reduce(concat, args)


We will need a function to print any frame to standard out. For this, we can create a ``pipe_node`` decorated function that, given a frame via ``fpn.PREDECESSOR_RETURN``, simply walks over the rows and prints the ``fpn.PN_INPUT`` defined ``pixel`` when a frame value is True, a space otherwise. Since this node returns the ``fpn.PREDECESSOR_RETURN`` unchanged, it can be used anywhere in an expression to view a frame mid-pipeline.


.. code-block:: python
    :class: copy-button

    @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def display(pixel_font_input, matrix):
        for row in matrix:
            for pixel in row:
                if pixel:
                    print(pixel_font_input.pixel, end="")
                else:
                    print(end=" ")
            print()
        return matrix


We have the tools now to define pipelines to produce the individual characters we need. We will define these in a dictionary, named ``chars``, so that we can map string characters to ``PipeNode`` expressions, pass them to ``concat``, and then pipe the results to ``display``. For brevity, we will not define a complete alphabet. For most characters the process involves taking the union of a number of lines (some shifted) and then flipping a few pixels. The font here is based on the Visitor font:

http://www.dafont.com/visitor.font


.. code-block:: python
    :class: copy-button

    chars = {
        "_" : frame,
        "." : frame | flip((2,4)),
        "p" : (
            union(
                frame | v_line,
                frame | h_line,
                frame | h_line | v_shift(2),
            )
            | flip((4,0), (4,1))
        ),
        "y" : (
            frame
            | h_line
            | v_shift(2)
            | flip((0,0), (0,1), (2,3), (2,4), (4,0), (4,1))
        ),
        "0" : union(
            frame | v_line,
            frame | v_line | h_shift(-1),
            frame | h_line,
            frame | h_line | v_shift(-1),
        ),
        "1" : frame | v_line | h_shift(2) | flip((1,0)),
        "2" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(4),
            )
            | flip((4, 1), (0, 3))
        ),
        "3" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(-1),
                frame | v_line | h_shift(4),
            )
            | flip((2, 2), (3, 2))
        ),
        "4" : (
            union(
                frame | h_line | v_shift(2),
                frame | v_line | h_shift(-1),
            )
            | flip((0, 0), (0, 1))
        ),
        "5" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(-1),
            )
            | flip((0, 1), (4, 3))
        ),
        "6" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(-1),
                frame | v_line,
            )
            | flip((4, 3))
        ),
        "7" : (
            (
                frame | h_line
            )
            | flip((2, 4), (2, 3), (3, 2), (4, 1))
        ),
        "8" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(-1),
                frame | v_line,
                frame | v_line | h_shift(4)
            )
        ),
        "9" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(-1),
                frame | v_line,
                frame | v_line | h_shift(4)
            )
            | flip((0, 3), (0, 4), (1, 4), (2, 4), (3, 4))
        ),
    }


We need a function to produce the final ``PipeNode`` expression. The ``msg_display_pipeline`` function, given a string message, will return the ``PipeNode`` expression combining ``concat`` and ``display``, where ``concat`` is called with PipeNode positional arguments, mapped from ``chars``, for each character passed in ``msg``. We map the "_" character for any characters not defined in ``chars``.


.. code-block:: python
    :class: copy-button

    def msg_display_pipeline(msg):
        get_char = lambda char: chars.get(char.lower(), chars["_"])
        return concat(*tuple(map(get_char, msg))) | display


Finally, we can define the outer-most application function, which will parse command-line arguments for ``pixel`` and ``scale`` with ``argparse.ArgumentParser``. The ``msg_display_pipeline`` function is called with the prepared ``msg`` string, returning ``f``, a ``PipeNode`` function configured to generate and display the ``msg`` as a banner. A ``PixelFontInput`` instance is created with the ``pixel`` and ``scale`` arguments received from the command line. At last, all *core callables* are called with the evocation of ``f`` with the ``__getitem__`` syntax, passing the ``PixelFontInput`` instance ``pixel_font_input``.


.. code-block:: python
    :class: copy-button

    def version_banner(args):

        p = argparse.ArgumentParser(
            description="Display the Python version in a banner",
        )
        p.add_argument(
            "--pixel",
            default="*",
            help="Set the character used for each pixel of the banner.",
        )
        p.add_argument(
            "--scale",
            default=1,
            type=int,
            help="Set the pixel scale for the banner.",
        )
        namespace = p.parse_args(args)
        assert len(namespace.pixel) == 1
        assert namespace.scale > 0

        msg = "py%s.%s.%s" % sys.version_info[:3]
        f = msg_display_pipeline(msg)

        pixel_font_input = PixelFontInput(pixel=namespace.pixel, scale=namespace.scale)
        f[pixel_font_input]


    if __name__ == "__main__":
        version_banner(sys.argv[1:])

Conclusion
==========

After going through this tutorial, you should now have an understanding of:

   - How to use ``fpn.PipeNode`` to do complex numpy array data pipline processing.

Here is all of the code examples we have seen so far:

.. code-block:: python
    :class: copy-button

    import argparse
    import functools
    import sys

    import numpy as np
    import function_pipe as fpn

    fpn.pipe_node = functools.partial(
        fpn.pipe_node,
        core_decorator=lambda f: f,
    )
    fpn.pipe_node_factory = functools.partial(
        fpn.pipe_node_factory,
        core_decorator=lambda f: f,
    )

    class PixelFontInput(fpn.PipeNodeInput):

        SHAPE = (5,5)

        def __init__(self, pixel="*", scale=1):
            super().__init__()
            self.scale = scale
            self.pixel = pixel

    @fpn.pipe_node(fpn.PN_INPUT)
    def frame(pixel_font_input):
        shape = tuple(v * pixel_font_input.scale for v in pixel_font_input.SHAPE)
        return np.zeros(shape=shape, dtype=bool)

    @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def v_line(pixel_font_input, matrix):
        matrix = matrix.copy()
        matrix[:, slice(0, pixel_font_input.scale)] = True
        return matrix

    @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def h_line(pixel_font_input, matrix):
        matrix = matrix.copy()
        matrix[slice(0, pixel_font_input.scale), :] = True
        return matrix

    @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN, fpn.PREDECESSOR_PN)
    def v_shift(pixel_font_input, matrix, predecessor, steps):
        if predecessor.unwrap == v_line.unwrap:
            raise Exception("cannot v_shift a v_line")
        return np.roll(matrix, pixel_font_input.scale * steps, axis=0)

    @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN, fpn.PREDECESSOR_PN)
    def h_shift(pixel_font_input, matrix, predecessor, steps):
        if predecessor.unwrap == h_line.unwrap:
            raise Exception("cannot h_shift an h_line")
        return np.roll(matrix, pixel_font_input.scale * steps, axis=1)

    @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def flip(pixel_font_input, matrix, *coords):
        matrix = matrix.copy()
        for coord in coords: # x, y pairs
            start = [i * pixel_font_input.scale for i in coord]
            end = [i + pixel_font_input.scale for i in start]
            iloc = slice(start[1], end[1]), slice(start[0], end[0])
            matrix[iloc] = ~matrix[iloc]
        return matrix

    @fpn.pipe_node_factory()
    def union(*args):
        return functools.reduce(np.logical_or, args)

    @fpn.pipe_node_factory()
    def intersect(*args):
        return functools.reduce(np.logical_and, args)

    @fpn.pipe_node_factory(fpn.PN_INPUT)
    def concat(pixel_font_input, *args):
        space = np.zeros(
            shape=(
                pixel_font_input.SHAPE[0] * pixel_font_input.scale,
                1 * pixel_font_input.scale
            ),
            dtype=bool,
        )
        concat = lambda x, y: np.concatenate((x, space, y), axis=1)
        return functools.reduce(concat, args)

    @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def display(pixel_font_input, matrix):
        for row in matrix:
            for pixel in row:
                if pixel:
                    print(pixel_font_input.pixel, end="")
                else:
                    print(end=" ")
            print()
        return matrix

    chars = {
        "_" : frame,
        "." : frame | flip((2,4)),
        "p" : (
            union(
                frame | v_line,
                frame | h_line,
                frame | h_line | v_shift(2),
            )
            | flip((4,0), (4,1))
        ),
        "y" : (
            frame
            | h_line
            | v_shift(2)
            | flip((0,0), (0,1), (2,3), (2,4), (4,0), (4,1))
        ),
        "0" : union(
            frame | v_line,
            frame | v_line | h_shift(-1),
            frame | h_line,
            frame | h_line | v_shift(-1),
        ),
        "1" : frame | v_line | h_shift(2) | flip((1,0)),
        "2" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(4),
            )
            | flip((4, 1), (0, 3))
        ),
        "3" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(-1),
                frame | v_line | h_shift(4),
            )
            | flip((2, 2), (3, 2))
        ),
        "4" : (
            union(
                frame | h_line | v_shift(2),
                frame | v_line | h_shift(-1),
            )
            | flip((0, 0), (0, 1))
        ),
        "5" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(-1),
            )
            | flip((0, 1), (4, 3))
        ),
        "6" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(-1),
                frame | v_line,
            )
            | flip((4, 3))
        ),
        "7" : (
            (
                frame | h_line
            )
            | flip((2, 4), (2, 3), (3, 2), (4, 1))
        ),
        "8" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(-1),
                frame | v_line,
                frame | v_line | h_shift(4)
            )
        ),
        "9" : (
            union(
                frame | h_line,
                frame | h_line | v_shift(2),
                frame | h_line | v_shift(-1),
                frame | v_line,
                frame | v_line | h_shift(4)
            )
            | flip((0, 3), (0, 4), (1, 4), (2, 4), (3, 4))
        ),
    }

    def msg_display_pipeline(msg):
        get_char = lambda char: chars.get(char.lower(), chars["_"])
        return concat(*tuple(map(get_char, msg))) | display

    def version_banner(args):

        p = argparse.ArgumentParser(
            description="Display the Python version in a banner",
        )
        p.add_argument(
            "--pixel",
            default="*",
            help="Set the character used for each pixel of the banner.",
        )
        p.add_argument(
            "--scale",
            default=1,
            type=int,
            help="Set the pixel scale for the banner.",
        )
        namespace = p.parse_args(args)
        assert len(namespace.pixel) == 1
        assert namespace.scale > 0

        msg = "py%s.%s.%s" % sys.version_info[:3]
        f = msg_display_pipeline(msg)

        pixel_font_input = PixelFontInput(pixel=namespace.pixel, scale=namespace.scale)
        f[pixel_font_input]


    if __name__ == "__main__":
        version_banner(sys.argv[1:])
