String Processing with FunctionNode and PipeNode
************************************************

Introduction
============

Simple examples of ``FunctionNode`` and ``PipeNode`` can be provided with string processing functions. While not serving any practical purpose, these examples demonstrate core features. Other usage examples will provide more practical demonstrations.



Importing function-pipe
=======================

Throughout these examples function-pipe will be imported as follows.

.. code-block:: python
    :class: copy-button

    import function_pipe as fpn

This assumes the function_pipe.py module has been installed in ``site-packages`` or is otherwise available via ``sys.path``.



FunctionNodes for Function Composition
======================================

FunctionNodes wrap callables. These callables can be lambdas, functions, or instances of callable classes. We can wrap them directly by calling ``FunctionNode`` or use ``FunctionNode`` as a decorator.

Using ``lambda`` callables for brevity, we can start with a number of simple functions that concatenate a string to an input string.

.. code-block:: python
    :class: copy-button

    a = fpn.FunctionNode(lambda s: s + "a")
    b = fpn.FunctionNode(lambda s: s + "b")
    c = fpn.FunctionNode(lambda s: s + "c")
    d = fpn.FunctionNode(lambda s: s + "d")
    e = fpn.FunctionNode(lambda s: s + "e")


With or without the ``FunctionNode`` decorator, we can call and compose these in Python with nested calls, such that the return of the inner function is the argument to the outer function.

.. code-block:: python
    :class: copy-button

    x = e(d(c(b(a("*")))))
    assert x == "*abcde"

This approach does not return a new function we can use repeatedly with different inputs. To do so, we can wrap the same nested calls in a ``lambda``. The *initial input* is the input provided to the resulting composed function.

.. code-block:: python
    :class: copy-button

    f = lambda x: e(d(c(b(a(x)))))
    assert f("*") == "*abcde"

While this works, it can be hard to maintain. By using FunctionNodes, we can make this composition more readable through it's linear ``>>`` or ``<<`` operators.

Both of these operators return a ``FunctionNode`` that, when called, pipes inputs to outputs (``>>``: left to right, ``<<``: left to right). As with the ``lambda`` example above, we can reuse the resulting ``FunctionNode`` with different inputs.

.. code-block:: python
    :class: copy-button

    f = a >> b >> c >> d >> e
    assert f("*") == "*abcde"
    assert f("?") == "?abcde"


Depending on your perspective, a linear presentation from left to right may not map well to the nested presentation initially given. The ``<<`` operator can be used to process from right to left:

.. code-block:: python
    :class: copy-button

    f = a << b << c << d << e
    assert f("*") == "*edcba"

And even though it is ill-advised on grounds of poor readability and unnecessary conceptual complexity, you can do bidirectional composition too:

.. code-block:: python
    :class: copy-button

    f = a >> b >> c << d << e
    assert f("*") == "*edabc"

The ``FunctionNode`` overloads standard binary and unary operators to produce new ``FunctionNodes`` that encapsulate operator operations. Operators can be mixed with composition to create powerful expressions.

.. code-block:: python
    :class: copy-button

    f = a >> (b * 4) >> (c + "___") >> d >> e
    assert f("*") == "*ab*ab*ab*abc___de"

We can create multiple FunctionNode expressions and combine them with operators and other compositions. Notice that the *initial input* "*" is made available to both *innermost* expressions, ``p`` and ``q``.

.. code-block:: python
    :class: copy-button

    p = c >> (b + "_") * 2
    q = d >> e * 2
    f = (p + q) * 2 + q
    assert f("*") == "*cb_*cb_*de*de*cb_*cb_*de*de*de*de"
    assert f("+") == "+cb_+cb_+de+de+cb_+cb_+de+de+de+de"


In the preceeding examples the functions took only the value of the *predecessor return* as their input. Each function thus has only one argument. Functions with additional arguments are much more useful.

As is common in approaches to function composition, we can partial multi-argument functions so as to compose them in a state where they only require the *predecessor return* as their input.

The ``FunctionNode`` exposes a ``partial`` method that simply calls ``functools.partial`` on the wrapped callable, and returns that new partialed function re-wrapped in a ``FunctionNode``.


.. code-block:: python
    :class: copy-button

    replace = fpn.FunctionNode(lambda s, src, dst: s.replace(src, dst))

    p = c >> (b + "_") * 2 >> replace.partial(src="b", dst="B$")
    q = d >> e * 2 >> replace.partial(src="d", dst="%D")
    f = (p + q) * 2 + q

    assert f("*") == "*cB$_*cB$_*%De*%De*cB$_*cB$_*%De*%De*%De*%De"



PipeNodes for Extended Function Composition
===========================================

At higher level of complexity, ``FunctionPipe`` can start to become difficult to understand or maintain. The ``PipeNode`` class (a subclass of ``FunctionNode``) and its associated decorators makes *extended function composition* practical, readable, and maintainable. Rather than using the ``>>`` or ``<<`` operators used by ``FunctionNode``, ``PipeNode`` uses only the ``|`` operator to express left-to-right composition.

We will build on the tutorial from earlier (LINK NEEDED), and now explore more complex string processing functions using ``PipeNode``.

Using the function ``a`` from before, we will instead create it as a ``PipeNode``, using the ``pipe_node`` decorator.

.. code-block:: python
    :class: copy-button

    a = fpn.pipe_node(fpn.PREDECESSOR_RETURN)(lambda s: s + "a")

Recall that PNs that receive ``fpn.PREDECESSOR_RETURN`` must have a preceding PN. In our case, we want an initial PN that receives an *initial input* from the user. We will do this by positionally binding ``fpn.PN_INPUT`` to the first argument.

.. code-block:: python
    :class: copy-button

    init = fpn.pipe_node(fpn.PN_INPUT)(lambda s: s)

Finally, we can generalize string concatenation with a ``cat`` function that, given an arbitrary string, concatenates it to its predecessor return value. Since this function takes an *expresion-level argument*, we must use the ``pipe_node_factory`` decorator.

.. code-block:: python
    :class: copy-button

    cat = fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)(lambda s, chars: s + chars)


Now we can create a pipeline expression that evaluates to a single function ``f``. In order to evaluate the pipeline, recall we must the ``__getitem__`` syntax with some initial input.

.. code-block:: python
    :class: copy-button

    f = init | a | cat("b") | cat("c")
    assert f["*"] == "*abc"
    assert f["+"] == "+abc"

Each node in a ``PipeNode`` expression has access to the ``fpn.PN_INPUT``. This can be used for many applications. A trivial application below replaces *initial input* characters found in the *predecessor return* with characters provided with the *expression-level argument* ``chars``.

.. code-block:: python
    :class: copy-button

    @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def replace_init(pni, s, chars):
        return s.replace(pni, chars)

    f = init | a | cat("b") | cat("c") * 2 | replace_init("+")
    assert f["*"] == "+abc+abc"


As already shown, a callable decorated with ``pipe_node_factory`` can take *expression-level arguments*. With a ``PipeNode`` expression, these arguments can be ``PipeNode`` expressions. The following function interleaves *expression-level arguments* with those of the *predecessor return* value.

.. code-block:: python
    :class: copy-button

    @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
    def interleave(s, chars):
        post = []
        for i, c in enumerate(s):
            post.append(c)
            post.append(chars[i % len(chars)])
        return "".join(post)

    h = init | cat("@@") | cat("__") * 2

    f = init | a | cat("b") | cat("c") * 3 | replace_init("+") | interleave(h)

    assert f["*"] == "+*a@b@c_+_a*b@c@+_a_b*c@"


We can break ``PipeNode`` expressions into pieces by storing and recalling results. This requires that the *initial input* is a ``PipeNodeInput`` or a subclass. The following ``PNI`` class exposes the ``__init__`` based ``chars`` argument as an instance attribute. Alternative designs for ``PipeNodeInput`` subclasses can provide a range of input data preparation. Since our *initial input* has changed, we need a new *innermost* node. The ``input_init`` node defined below simply returns the ``chars`` attribute from the ``PNI`` instance passed as key-word argument ``fpn.PN_INPUT``.

The function-pipe module provides ``store`` and ``recall`` nodes. The ``store`` node stores a predecessor value. The ``recall`` node returns a stored value as an output later in the expression. A ``recall`` node, for example, can be used as an argument to ``pipe_node_factory`` functions. The ``call`` ``PipeNode``, also provided in the function-pipe module, will call any number of passed ``PipeNode`` expressions in sequence.

.. code-block:: python
    :class: copy-button

    class PNI(fpn.PipeNodeInput):
        def __init__(self, chars):
            super().__init__()
            self.chars = chars

    @fpn.pipe_node(fpn.PN_INPUT)
    def input_init(pni):
        return pni.chars

    p = input_init | cat("www") | fpn.store("p")
    q = input_init | cat("@@") | cat("__") * 2 | fpn.store("q")
    r = (
        input_init
        | a
        | cat(fpn.recall("p"))
        | cat("c") * 3
        | interleave(fpn.recall("q"))
    )

    f = fpn.call(p, q, r)
    pni = PNI("x")

    assert f[pni] == "xxa@x@w_w_wxc@x@a_x_wxw@w@c_x_axx@w@w_w_cx"

While these string processors do not do anything useful, they demonstrate common approaches in working with ``FunctionNode`` and ``PipeNode``.



Conclusion
==========

After going through this tutorial, you should now have an understanding of:

   - How to use ``fpn.FunctionNode`` for function composition
   - The directionality of ``fpn.FunctionPipe`` (i.e. ``>>`` and ``<<``)
   - How to partial expression-level arguments into ``fpn.FunctionPipe``
   - The ``fpn.pipe_node`` decorator, and when to use it
   - The ``fpn.pipe_node_factory`` decorator, and when to use it
   - How to use ``fpn.PipeNode`` for function composition

Here is all of the code examples we have seen so far:

.. code-block:: python
    :class: copy-button

    import function_pipe as fpn

    a = fpn.FunctionNode(lambda s: s + "a")
    b = fpn.FunctionNode(lambda s: s + "b")
    c = fpn.FunctionNode(lambda s: s + "c")
    d = fpn.FunctionNode(lambda s: s + "d")
    e = fpn.FunctionNode(lambda s: s + "e")

    x = e(d(c(b(a("*")))))
    assert x == "*abcde"

    # -------------------------------------------------------------------------

    f = lambda x: e(d(c(b(a(x)))))
    assert f("*") == "*abcde"

    # -------------------------------------------------------------------------

    f = a >> b >> c >> d >> e
    assert f("*") == "*abcde"
    assert f("?") == "?abcde"

    # -------------------------------------------------------------------------

    f = a << b << c << d << e
    assert f("*") == "*edcba"

    # -------------------------------------------------------------------------

    f = a >> b >> c << d << e
    assert f("*") == "*edabc"

    # -------------------------------------------------------------------------

    f = a >> (b * 4) >> (c + "___") >> d >> e
    assert f("*") == "*ab*ab*ab*abc___de"

    # -------------------------------------------------------------------------

    p = c >> (b + "_") * 2
    q = d >> e * 2
    f = (p + q) * 2 + q
    assert f("*") == "*cb_*cb_*de*de*cb_*cb_*de*de*de*de"
    assert f("+") == "+cb_+cb_+de+de+cb_+cb_+de+de+de+de"

    # -------------------------------------------------------------------------

    replace = fpn.FunctionNode(lambda s, src, dst: s.replace(src, dst))

    p = c >> (b + "_") * 2 >> replace.partial(src="b", dst="B$")
    q = d >> e * 2 >> replace.partial(src="d", dst="%D")
    f = (p + q) * 2 + q

    assert f("*") == "*cB$_*cB$_*%De*%De*cB$_*cB$_*%De*%De*%De*%De"

    # -------------------------------------------------------------------------

    a = fpn.pipe_node(fpn.PREDECESSOR_RETURN)(lambda s: s + "a")

    init = fpn.pipe_node(fpn.PN_INPUT)(lambda s: s)

    cat = fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)(lambda s, chars: s + chars)

    f = init | a | cat("b") | cat("c")
    assert f["*"] == "*abc"
    assert f["+"] == "+abc"

    # -------------------------------------------------------------------------

    @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    def replace_init(pni, s, chars):
        return s.replace(pni, chars)

    f = init | a | cat("b") | cat("c") * 2 | replace_init("+")
    assert f["*"] == "+abc+abc"

    # -------------------------------------------------------------------------

    @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
    def interleave(s, chars):
        post = []
        for i, c in enumerate(s):
            post.append(c)
            post.append(chars[i % len(chars)])
        return "".join(post)

    h = init | cat("@@") | cat("__") * 2

    f = init | a | cat("b") | cat("c") * 3 | replace_init("+") | interleave(h)

    assert f["*"] == "+*a@b@c_+_a*b@c@+_a_b*c@"

    # -------------------------------------------------------------------------

    class PNI(fpn.PipeNodeInput):
        def __init__(self, chars):
            super().__init__()
            self.chars = chars

    @fpn.pipe_node(fpn.PN_INPUT)
    def input_init(pni):
        return pni.chars

    p = input_init | cat("www") | fpn.store("p")
    q = input_init | cat("@@") | cat("__") * 2 | fpn.store("q")
    r = (
        input_init
        | a
        | cat(fpn.recall("p"))
        | cat("c") * 3
        | interleave(fpn.recall("q"))
    )

    f = fpn.call(p, q, r)
    pni = PNI("x")

    assert f[pni] == "xxa@x@w_w_wxc@x@a_x_wxw@w@c_x_axx@w@w_w_cx"
