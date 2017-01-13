


import function_pipe as fpn


# create string lambda function

# can have an inteleave() processor
# can have a replace() process

# both can be fed from other string processors


def fn():

    # givne a bunch of callables (lambdas, functions, callable instances), we can easily compose them

    a = fpn.FunctionNode(lambda s: s + 'a')
    b = fpn.FunctionNode(lambda s: s + 'b')
    c = fpn.FunctionNode(lambda s: s + 'c')
    d = fpn.FunctionNode(lambda s: s + 'd')
    e = fpn.FunctionNode(lambda s: s + 'e')

    x = e(d(c(b(a('*')))))
    assert x == '*abcde'

    f = lambda x: e(d(c(b(a(x)))))
    assert f('*') == '*abcde'


    f = a >> b >> c >> d >> e
    print(f('*'))
    assert f('*') == '*abcde'

    print(f('?'))
    assert f('?') == '?abcde'


    # we can compose from left to right
    f = a << b << c << d << e
    print(f('*'))
    assert f('*') == '*edcba'


    # or bi-directionally (though not reccommended)
    f = a >> b >> c << d << e
    print(f('*'))
    assert f('*') == '*edabc'


    # we can use operators to produce new, composable FNs; operators perform based on the underlying operator overloading for the piped type. We can integreate constants too.

    f = a >> (b * 4) >> (c + '___') >> d >> e
    print(f('*'))
    assert f('*') == '*ab*ab*ab*abc___de'


    # we can combine sub-pipes with operators and use pipes and pipe-expressions in repeatedly

    p = c >> (b + '_') * 2
    q = d >> e * 2
    f = (p + q) * 2 + q

    print(f('*'))
    assert f('*') == '*cb_*cb_*de*de*cb_*cb_*de*de*de*de'

    print(f('+'))
    assert f('+') == '+cb_+cb_+de+de+cb_+cb_+de+de+de+de'


    # we can intergrate functions that take arguments through partialling, using either args or kwargs

    replace = fpn.FunctionNode(lambda s, src, dst: s.replace(src, dst))

    p = c >> (b + '_') * 2 >> replace.partial(src='b', dst='B$')
    q = d >> e * 2 >> replace.partial(src='d', dst='%D')
    f = (p + q) * 2 + q

    print(f('*'))
    assert f('*') == '*cB$_*cB$_*%De*%De*cB$_*cB$_*%De*%De*%De*%De'



def pn():
    # if we are willing to adjust our functions about, we can do even more with PipeNodes. PipeNodes have a protocal; they have to be called in a certain way, and have certain expectations regarding arguments. PipeNodes are created through decorators. Decorated functions receive PipeNode qwargs

    a = fpn.pipe_node(lambda **kwargs: kwargs[fpn.PREDECESSOR_RETURN] + 'a')

    # another way of doing the same thing
    @fpn.pipe_node
    @fpn.pipe_kwarg_bind(fpn.PREDECESSOR_RETURN)
    def a(s):
        return s + 'a'

    # capturing the initial input is not automatic
    init = fpn.pipe_node(lambda **kwargs: kwargs[fpn.PN_INPUT])

    @fpn.pipe_node_factory
    def cat(chars, **kwargs):
        return kwargs[fpn.PREDECESSOR_RETURN] + chars

    f = init | a | cat('b') | cat('c')


    print(f(pn_input='*'))
    assert f(pn_input='*') == '*abc'

    # can also be done as the following

    print(f(**{fpn.PN_INPUT: '*'}))
    assert f(**{fpn.PN_INPUT: '+'}) == '+abc'

    print(f['*'])
    assert f['*'] == '*abc'


    # with pipenodes, all nodes have access to the PN_INPUT through the common kwargs

    @fpn.pipe_node_factory
    def replace_init(chars, **kwargs):
        return kwargs[fpn.PREDECESSOR_RETURN].replace(kwargs[fpn.PN_INPUT], chars)


    f = init | a | cat('b') | cat('c') * 2 | replace_init('+')
    print(f['*'])
    assert f['*'] == '+abc+abc'


    # as already shown pipe-nodes can take arguments; those arguments can be pipe nodes themselves

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

    print(f['*'])
    assert f['*'] == '+*a@b@c_+_a*b@c@+_a_b*c@'

    # if we want to use a PN expression more than once, we can store it and recall it later. To do so, we need to use a PipeNodeInput or itse subclass

    class Input(fpn.PipeNodeInput):
        def __init__(self, chars):
            super().__init__()
            self.chars = chars

    # we nee dot change our init function to read the chars attr
    input_init = fpn.pipe_node(lambda **kwargs: kwargs[fpn.PN_INPUT].chars)

    p = input_init | cat('www') | fpn.store('p')
    q = input_init | cat('@@') | cat('__') * 2 | fpn.store('q')
    r = input_init | a | cat(fpn.recall('p')) | cat('c') * 3 | interleave(fpn.recall('q'))
    f = fpn.call(p, q, r)

    pni = Input('x')
    print(f[pni])

    pni = Input('x') # must create a new one
    assert f[pni] == 'xxa@x@w_w_wxc@x@a_x_wxw@w@c_x_axx@w@w_w_cx'




if __name__ == '__main__':
    fn()
    pn()

