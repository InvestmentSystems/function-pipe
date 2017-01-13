

Usage with String Processing
==================================

Simple examples of ``FunctionNode`` and ``PipeNode`` can be provided with string processing functions. While not serving any practical purpose, these examples demonstrate core features. Other usage examples will provide more practical demonstrations.


Importing function-pipe
------------------------------------------

Throughout these examples function-pipe will be imported as follows.

.. code-block:: python

    import function_pipe as fpn

This assumes the function_pipe.py module has been installed in ``site-packages`` or is otherwise available via ``sys.path``.



FunctionNodes for Function Composition
------------------------------------------

FunctionNodes wrap callables. These callables can be lambdas, functions, or callable instances. We can wrap them directly by calling ``FunctionNode`` or use ``FunctionNode`` as a decorator.

Using ``lambda`` callables for brevity, we can start with a number of simple functions that concatenate a string to an input string.

.. code-block:: python

    a = fpn.FunctionNode(lambda s: s + 'a')
    b = fpn.FunctionNode(lambda s: s + 'b')
    c = fpn.FunctionNode(lambda s: s + 'c')
    d = fpn.FunctionNode(lambda s: s + 'd')
    e = fpn.FunctionNode(lambda s: s + 'e')


With or without the ``FunctionNode`` decorator, we can call and compose these in Python with nested calls, such that the return of the inner function is the argument to the outer function.

.. code-block:: python

    x = e(d(c(b(a('*')))))
    assert x == '*abcde'

This approach does not return a new function we can use repeatedly with different inputs. To do so, we can wrap the same nested calls in a ``lambda``. The *initial input* is the input provided to the resulting composed function.

.. code-block:: python

    f = lambda x: e(d(c(b(a(x)))))
    assert f('*') == '*abcde'

Maintaining lots of functions in a nested presentation is unwieldy. As FunctionNodes, we can make the composition linear (and thus readable) by using the ``>>`` or ``<<`` operators.

The ``>>`` returns a ``FunctionNode`` that, when called, pipes inputs to outputs from left to right. As with the ``lambda`` example above, we can reuse the resulting ``FunctionNode`` with different inputs.

.. code-block:: python

    f = a >> b >> c >> d >> e
    assert f('*') == '*abcde'
    assert f('?') == '?abcde'


Depending on your perspective, a linear presentation from left to right may not map well to the nested presentation initially given. The ``<<`` operator can be used to process from right to left:

.. code-block:: python

    f = a << b << c << d << e
    assert f('*') == '*edcba'

And even though it is ill-advised on grounds of poor readability and unnecessary conceptual complexity, you can do bidirectional composition too:

.. code-block:: python

    f = a >> b >> c << d << e
    assert f('*') == '*edabc'

The ``FunctionNode`` overloads standard binary and unary operators to produce new ``FunctionNodes`` that encapsulate operator operations. Operators can be mixed with composition to create powerful expressions.

.. code-block:: python

    f = a >> (b * 4) >> (c + '___') >> d >> e
    assert f('*') == '*ab*ab*ab*abc___de'

We can create multiple FunctionNode expressions and combine them with operators and other compositions. Notice that the *initial input* "*" is made available to both *innermost* expressions, ``p`` and ``q``.

.. code-block:: python

    p = c >> (b + '_') * 2
    q = d >> e * 2
    f = (p + q) * 2 + q
    assert f('*') == '*cb_*cb_*de*de*cb_*cb_*de*de*de*de'
    assert f('+') == '+cb_+cb_+de+de+cb_+cb_+de+de+de+de'


In the preceeding examples the functions took only the value of the *predecessor return* as their input. Each function thus has only one argument. Functions with additional arguments are much more useful.

As is common in approaches to function composition, we can partial (or curry in other applications) multi-argument functions so as to compose them in a state where they only require the *predecessor return* as their input.

The ``FunctionNode`` exposes a ``partial`` method that simply calls ``functools.partial`` on the wrapped callable, and returns that new partialed function re-wrapped in a ``FunctionNode``.


.. code-block:: python

    replace = fpn.FunctionNode(lambda s, src, dst: s.replace(src, dst))

    p = c >> (b + '_') * 2 >> replace.partial(src='b', dst='B$')
    q = d >> e * 2 >> replace.partial(src='d', dst='%D')
    f = (p + q) * 2 + q

    print(f('*'))
    assert f('*') == '*cB$_*cB$_*%De*%De*cB$_*cB$_*%De*%De*%De*%De'



PipeNodes for Extended Function Composition
---------------------------------------------

Function composition as presented above becomes unwieldy at greater levels of complexity. The ``PipeNode`` class (a subclass of ``FunctionNode``) and its associated decorators makes *extended function composition* practical, readable, and maintainable. Rather than using the ``>>`` or ``<<`` operators used by ``FunctionNode``, ``PipeNode`` uses only the ``|`` operator to express left-to-right composition.

Unlike with ``FunctionNode`` usage, the ``PipeNode`` class is rarely called directly to create instances. Rather, two decorators, ``pipe_node`` and ``pipe_node_factory``, are applied to *core callables*. These decorators embed the callable in a two- (or three-) part call structure, each call returning a ``PipeNode`` instance in one of three sequential call states: ``PipeNode.FACTORY``, ``PipeNode.EXPRESSION``, and ``PipeNode.PROCESS``. Generally, using the correct decorator insures that you do not need to consider underling ``PipeNode`` states.

The *PipeNode protocol* requires all *core callables* wrapped by ``PipeNode`` decorators to take at least ``**kwargs``; PipeNode key-word arguments ``fpn.PREDECESSOR_RETURN``, ``fpn.PREDECESSOR_PN``, and ``fpn.PN_INPUT`` are, as appropriate, passed as key-word arguments by the decorators to the *core callable*.

A function analogous to the above ``FunctionNode`` ``a``, now as a ``PipeNode``, can be defined in a few different ways. The function can read ``fpn.PREDECESSOR_RETURN`` from the key-word arguments, or a positional-argument function can have ``PipeNode`` key-word arguments bound to positional arguments with the ``pipe_kwarg_bind`` decorator.

.. code-block:: python

    a = fpn.pipe_node(lambda **kwargs: kwargs[fpn.PREDECESSOR_RETURN] + 'a')

    @fpn.pipe_node
    @fpn.pipe_kwarg_bind(fpn.PREDECESSOR_RETURN)
    def a(s):
        return s + 'a'

The ``PipeNode`` decorators deliver the *initial input* to every PipeNode as the key-word argument ``fpn.PN_INPUT``. The *innermost* ``PipeNode`` in an expression does not have a predecessor, and thus receives only the ``fpn.PN_INPUT`` key-word argument. All other PipeNodes receive all three key-word arguments, ``fpn.PREDECESSOR_RETURN``, ``fpn.PREDECESSOR_PN``, and ``fpn.PN_INPUT``.

For this reason, the *innermost* PipeNode can only access ``fpn.PN_INPUT``. We can define an *innermost* function that simply passes on the *initial input*, ``fpn.PN_INPUT``, as follows:

.. code-block:: python

    init = fpn.pipe_node(lambda **kwargs: kwargs[fpn.PN_INPUT])

Finally, we can generalize string concatenation with a ``cat`` function that, given an arbitrary string, concatenates it to its predecessor return value. Since this function takes an expresion-level argument, we use the ``pipe_node_factory`` decorator.

.. code-block:: python

    @fpn.pipe_node_factory
    def cat(chars, **kwargs):
        return kwargs[fpn.PREDECESSOR_RETURN] + chars


Now we can create an expression that evaluates to a single function ``f``. The ``init`` node, in the *innermost* position, is used to pass on the ``fpn.PN_INPUT`` to the subsequent nodes. Note that the resulting function ``f`` has to be called the key-word argument ``fpn.PN_INPUT``.

.. code-block:: python

    f = init | a | cat('b') | cat('c')
    assert f(**{fpn.PN_INPUT: '*'}) == '*abc'
    assert f(**{fpn.PN_INPUT: '+'}) == '+abc'

We can avoid calling function ``f`` with a key-word argument by using the ``__getitem__`` syntax, overridden here to call the passed argument as the ``fpn.PN_INPUT``.

.. code-block:: python

    assert f['*'] == '*abc'


Each node in a ``PipeNode`` expression has access to the ``fpn.PN_INPUT``. This can be used for many applications. A trivial application below replaces *initial input* characters found in the *predecessor return* with characters provided with the *expression-level argument* ``chars``.

.. code-block:: python

    @fpn.pipe_node_factory
    def replace_init(chars, **kwargs):
        return kwargs[fpn.PREDECESSOR_RETURN].replace(kwargs[fpn.PN_INPUT], chars)

    f = init | a | cat('b') | cat('c') * 2 | replace_init('+')
    assert f['*'] == '+abc+abc'


As already shown, a callable decorated with ``pipe_node_factory`` can take *expression-level arguments*. With a ``PipeNode`` expression, these arguments can be ``PipeNode`` expressions. The following function interleaves *expression-level arguments* with those of the *predecessor return* value.

.. code-block:: python

    @fpn.pipe_node_factory
    def interleave(chars, **kwargs):
        pred = kwargs[fpn.PREDECESSOR_RETURN]
        post = []
        for i, c in enumerate(pred):
            post.append(c)
            post.append(chars[i % len(chars)])
        return ''.join(post)

    h = init | cat('@@') | cat('__') * 2

    f = init | a | cat('b') | cat('c') * 3 | replace_init('+') | interleave(h)

    assert f['*'] == '+*a@b@c_+_a*b@c@+_a_b*c@'


We can break ``PipeNode`` expressions into pieces by storing and recalling results. This requires that the *initial input* is a ``PipeNodeInput`` or a subclass. The following ``Input`` class exposes the ``__init__`` based ``chars`` argument as an instance attribute. Alternative designs for ``PipeNodeInput`` subclasses can provide a range of input data preparation. Since our *initial input* has changed, we need a new *innermost* node. The ``input_init`` node defined below simply returns the ``chars`` attribute from the ``Input`` instance passed as key-word argument ``fpn.PN_INPUT``.

The function-pipe module provides ``store`` and ``recall`` nodes. The ``store`` node stores a predecessor value. The ``recall`` node returns a stored value as an output later in the expression. A ``recall`` node, for example, can be used as an argument to ``pipe_node_factory`` functions. The ``call`` ``PipeNode``, also provided in the function-pipe module, will call any number of passed ``PipeNode`` expressions in sequence.

.. code-block:: python

    class Input(fpn.PipeNodeInput):
        def __init__(self, chars):
            super().__init__()
            self.chars = chars

    @fpn.pipe_node
    def input_init(**kwargs):
        return kwargs[fpn.PN_INPUT].chars

    p = input_init | cat('www') | fpn.store('p')
    q = input_init | cat('@@') | cat('__') * 2 | fpn.store('q')
    r = (input_init | a | cat(fpn.recall('p')) | cat('c') * 3
            | interleave(fpn.recall('q')))

    f = fpn.call(p, q, r)
    pni = Input('x')

    assert f[pni] == 'xxa@x@w_w_wxc@x@a_x_wxw@w@c_x_axx@w@w_w_cx'


While these string processors do not do anything useful, they demonstrate common approaches in working with ``FunctionNode`` and ``PipeNode``.



