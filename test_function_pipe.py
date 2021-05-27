# pylint: disable=no-method-argument, no-self-argument, too-many-function-args
# pylint: disable=invalid-sequence-index, missing-kwoa, unsubscriptable-object
# pylint: disable=no-value-for-parameter, invalid-unary-operand-type
# pylint: disable=pointless-string-statement, pointless-statement
import functools
import types
import unittest

import function_pipe as fpn

class TestUnit(unittest.TestCase):
    def test_basic_expressions_a(self):
        class TestInput(fpn.PipeNodeInput):
            def get_origin(self):
                return 32

        @fpn.pipe_node
        def proc_a(**kwargs):
            gmi = kwargs[fpn.PN_INPUT]
            post = gmi.get_origin()
            return post

        @fpn.pipe_node_factory
        def proc_b(scalar, **kwargs):
            pred = kwargs[fpn.PREDECESSOR_RETURN]
            post = pred * scalar
            return post

        @fpn.pipe_node
        def proc_c(**kwargs):
            pred = kwargs[fpn.PREDECESSOR_RETURN]
            post = pred / 10000
            return post

        @fpn.pipe_node
        def proc_d(**kwargs):
            pred = kwargs[fpn.PREDECESSOR_RETURN]
            return pred


        a = proc_a | proc_b(scalar=3) | proc_d | proc_b(scalar=.5) | proc_c
        b = (proc_a * 30)
        c = proc_a | proc_b(scalar=b)
        d = proc_a | -proc_b(3)

        f = a + b

        gmi = TestInput()

        post = f(pn_input=gmi)
        self.assertEqual(post, 960.0048)

        post = a(pn_input=gmi)
        self.assertEqual(post, 0.0048)

        post = b(pn_input=gmi)
        self.assertEqual(post, 960)

        post = c(pn_input=gmi)
        self.assertEqual(post, 30720)

        post = d(pn_input=gmi)
        self.assertEqual(post, -96)

        # proc a can reside in non first position

        f = proc_a | proc_a | proc_a
        post = f(pn_input=gmi)
        self.assertEqual(post, 32)


        f = proc_a | proc_a | proc_a
        post = f(pn_input=gmi)
        self.assertEqual(post, 32)

    def test_basic_expressions_b(self):
        @fpn.pipe_node
        def foo1(**kwargs):
            return 12

        @fpn.pipe_node
        def bar1(**kwargs):
            return 13

        self.assertEqual(12, (bar1 | foo1)[None])
        self.assertEqual(13, (foo1 | bar1)[None])

    def test_basic_expressions_c(self):
        @fpn.pipe_node(fpn.PN_INPUT)
        def foo2(pni):
            return pni

        @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def bar2(pni, prev):
            return pni + prev

        self.assertEqual(27 * 2, (foo2 | bar2)[27])

    def test_basic_expressions_d(self):
        @fpn.pipe_node(fpn.PN_INPUT)
        def foo3(pni):
            return pni

        @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def bar3(pni, prev):
            return pni + prev

        self.assertEqual(27 * 2, (foo3 | bar3)[27])

    def test_basic_expressions_e(self):
        @fpn.pipe_node_factory
        def foo4(arg, **kwargs):
            return 12 + arg

        @fpn.pipe_node_factory
        def bar4(arg, **kwargs):
            return 13 - arg

        self.assertEqual(12 + 7, (bar4(6) | foo4(7))[None])
        self.assertEqual(13 - 6, (foo4(7) | bar4(6))[None])

    def test_basic_expressions_f(self):
        @fpn.pipe_node_factory(fpn.PN_INPUT)
        def foo5(pni, arg):
            return pni + arg

        @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def bar5(pni, prev, arg):
            return pni + prev - arg

        self.assertEqual(27 * 2 + 8 - 2, (foo5(8) | bar5(2))[27])

    def test_basic_expressions_g(self):
        @fpn.pipe_node_factory(fpn.PN_INPUT)
        def foo6(pni, arg):
            return pni + arg

        @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def bar6(pni, prev, arg):
            return pni + prev - arg

        self.assertEqual(27 * 2 + 8 - 2, (foo6(8) | bar6(2))[27])

    def test_methods_defined_on_classes(self):
        class C:
            STATE = 'C class state'

            def __init__(self, state) -> None:
                self.state = f'C {state}'

            # classmethods

            @fpn.classmethod_pipe_node
            def cls_node(cls, **kwargs):
                return cls.STATE

            @fpn.classmethod_pipe_node(fpn.PN_INPUT)
            def cls_node_bind(cls, pni):
                return pni, cls.STATE

            @fpn.classmethod_pipe_node_factory
            def cls_node_factory(cls, factory_val, **kwargs):
                return f"{cls.__name__} factory_val='{factory_val}'"

            @fpn.classmethod_pipe_node_factory(fpn.PN_INPUT)
            def cls_node_factory_bind(cls, pni, factory_val):
                return pni, f"{cls.__name__} factory_val='{factory_val}'"

            # staticmethods

            @fpn.staticmethod_pipe_node
            def staticmethod_node(**kwargs):
                return 'STATIC'

            @fpn.staticmethod_pipe_node(fpn.PN_INPUT)
            def staticmethod_node_bind(pni):
                return pni, 'STATIC'

            @fpn.staticmethod_pipe_node_factory
            def staticmethod_node_factory(factory_val, **kwargs):
                return f"STATIC factory_val='{factory_val}'"

            @fpn.staticmethod_pipe_node_factory(fpn.PN_INPUT)
            def staticmethod_node_factory_bind(pni, factory_val):
                return pni, f"STATIC factory_val='{factory_val}'"

            # namespace methods. These should fail if called from an instance.

            @fpn.pipe_node
            def namespace_node(**kwargs):
                return 'NAMESPACE'

            @fpn.pipe_node(fpn.PN_INPUT)
            def namespace_node_bind(pni):
                return pni, 'NAMESPACE'

            @fpn.pipe_node_factory
            def namespace_node_factory(factory_val, **kwargs):
                return f"NAMESPACE factory_val='{factory_val}'"

            @fpn.pipe_node_factory(fpn.PN_INPUT)
            def namespace_node_factory_bind(pni, factory_val):
                return pni, f"NAMESPACE factory_val='{factory_val}'"

            # self methods

            @fpn.pipe_node
            def self_node(self, **kwargs):
                return self.state

            @fpn.pipe_node(fpn.PN_INPUT)
            def self_node_bind(self, pni):
                return pni, self.state

            @fpn.pipe_node_factory
            def self_node_factory(self, factory_val, **kwargs):
                return f"SELF factory_val='{factory_val}'"

            @fpn.pipe_node_factory(fpn.PN_INPUT)
            def self_node_factory_bind(self, pni, factory_val):
                return pni, f"SELF factory_val='{factory_val}'"

        class D(C):
            STATE = 'D class state'

            def __init__(self, state) -> None:
                self.state = f'D {state}'

        d = D('self state')
        pni = 'PNI'

        self.assertEqual(d.cls_node[pni], 'D class state')
        self.assertEqual(D.cls_node[pni], 'D class state')
        self.assertEqual(C.cls_node[pni], 'C class state')
        self.assertEqual(d.cls_node_bind[pni], (pni, 'D class state'))
        self.assertEqual(D.cls_node_bind[pni], (pni, 'D class state'))
        self.assertEqual(C.cls_node_bind[pni], (pni, 'C class state'))
        self.assertEqual(d.cls_node_factory('val')[pni], "D factory_val='val'")
        self.assertEqual(D.cls_node_factory('val')[pni], "D factory_val='val'")
        self.assertEqual(C.cls_node_factory('val')[pni], "C factory_val='val'")
        self.assertEqual(d.cls_node_factory_bind('val')[pni], (pni, "D factory_val='val'"))
        self.assertEqual(D.cls_node_factory_bind('val')[pni], (pni, "D factory_val='val'"))
        self.assertEqual(C.cls_node_factory_bind('val')[pni], (pni, "C factory_val='val'"))

        self.assertEqual(d.staticmethod_node[pni], 'STATIC')
        self.assertEqual(D.staticmethod_node[pni], 'STATIC')
        self.assertEqual(C.staticmethod_node[pni], 'STATIC')
        self.assertEqual(d.staticmethod_node_bind[pni], (pni, 'STATIC'))
        self.assertEqual(D.staticmethod_node_bind[pni], (pni, 'STATIC'))
        self.assertEqual(C.staticmethod_node_bind[pni], (pni, 'STATIC'))
        self.assertEqual(d.staticmethod_node_factory('val')[pni], "STATIC factory_val='val'")
        self.assertEqual(D.staticmethod_node_factory('val')[pni], "STATIC factory_val='val'")
        self.assertEqual(C.staticmethod_node_factory('val')[pni], "STATIC factory_val='val'")
        self.assertEqual(d.staticmethod_node_factory_bind('val')[pni], (pni, "STATIC factory_val='val'"))
        self.assertEqual(D.staticmethod_node_factory_bind('val')[pni], (pni, "STATIC factory_val='val'"))
        self.assertEqual(C.staticmethod_node_factory_bind('val')[pni], (pni, "STATIC factory_val='val'"))

        self.assertEqual(d.namespace_node[pni], 'NAMESPACE')
        self.assertEqual(D.namespace_node[pni], 'NAMESPACE')
        self.assertEqual(C.namespace_node[pni], 'NAMESPACE')
        self.assertEqual(d.namespace_node_bind[pni], (pni, 'NAMESPACE'))
        self.assertEqual(D.namespace_node_bind[pni], (pni, 'NAMESPACE'))
        self.assertEqual(C.namespace_node_bind[pni], (pni, 'NAMESPACE'))
        self.assertEqual(d.namespace_node_factory('val')[pni], "NAMESPACE factory_val='val'")
        self.assertEqual(D.namespace_node_factory('val')[pni], "NAMESPACE factory_val='val'")
        self.assertEqual(C.namespace_node_factory('val')[pni], "NAMESPACE factory_val='val'")
        self.assertEqual(d.namespace_node_factory_bind('val')[pni], (pni, "NAMESPACE factory_val='val'"))
        self.assertEqual(D.namespace_node_factory_bind('val')[pni], (pni, "NAMESPACE factory_val='val'"))
        self.assertEqual(C.namespace_node_factory_bind('val')[pni], (pni, "NAMESPACE factory_val='val'"))

        self.assertEqual(d.self_node[pni], 'D self state')
        self.assertEqual(d.self_node_bind[pni], (pni, 'D self state'))
        self.assertEqual(d.self_node_factory('val')[pni], "SELF factory_val='val'")
        self.assertEqual(d.self_node_factory_bind('val')[pni], (pni, "SELF factory_val='val'"))

        '''Custom ValueError indicating an unbound self-method was called without an instance'''
        with self.assertRaises(ValueError):
            D.self_node[pni]

        with self.assertRaises(ValueError):
            C.self_node[pni]

        '''Normal errors about a missing required positional arg, which makes sense, since self wasn't passed in'''
        with self.assertRaises(TypeError):
            D.self_node_bind[pni]

        with self.assertRaises(TypeError):
            C.self_node_bind[pni]

        with self.assertRaises(TypeError):
            D.self_node_factory('val')[pni]

        with self.assertRaises(TypeError):
            C.self_node_factory('val')[pni]

        with self.assertRaises(TypeError):
            D.self_node_factory_bind('val')[pni]

        with self.assertRaises(TypeError):
            C.self_node_factory_bind('val')[pni]

    def test_bound_and_unbound_pipe_nodes(self):
        @fpn.pipe_node
        def pn_unbound(**kwargs):
            return kwargs[fpn.PN_INPUT]

        @fpn.pipe_node(fpn.PN_INPUT)
        def pn_bound(pni):
            return pni

        @fpn.pipe_node_factory
        def pnf_unbound(factory_val, **kwargs):
            return kwargs[fpn.PN_INPUT], factory_val

        @fpn.pipe_node_factory(fpn.PN_INPUT)
        def pnf_bound(pni, factory_val):
            return pni, factory_val

        pni = 'PNI'

        self.assertEqual(pn_unbound[pni], pn_bound[pni])
        self.assertEqual(pnf_unbound('fval')[pni], pnf_bound('fval')[pni])

    def test_complex_pipeline(self):
        @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
        def multiply(prev, val):
            return prev * val

        @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def add_pni(pni, prev):
            return prev + pni.value

        @fpn.pipe_node
        def init(**kwargs):
            return 12

        @fpn.pipe_node_factory
        def func(arg1, arg2, **kwargs):
            return (kwargs[fpn.PN_INPUT].value * arg1) + (kwargs[fpn.PREDECESSOR_RETURN] / arg2)

        class Operations:
            DELTA = 0.23

            def __init__(self, factor):
                self.factor = factor

            @fpn.classmethod_pipe_node
            def add_delta(cls, **kwargs):
                return cls.DELTA + kwargs[fpn.PREDECESSOR_RETURN]

            @fpn.staticmethod_pipe_node_factory(fpn.PREDECESSOR_RETURN)
            def bound_prev(prev, lower, upper):
                return max(min(prev, lower), upper)

            @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
            def adj_by_factor(self, pni, prev):
                return self.factor * (pni.value - prev)

        op = Operations(23.717)

        expr = (init |
                add_pni |
                fpn.store('A') |
                multiply(-8) |
                func(13, 7) |
                fpn.store('B') |
                op.add_delta |
                Operations.add_delta |
                op.bound_prev(0, 100) |
                fpn.store('C') |
                multiply(130.2) |
                Operations.bound_prev(100, 200) |
                op.adj_by_factor |
                fpn.store('D') |
                fpn.recall('A') |
                op.adj_by_factor |
                fpn.recall('B') |
                op.adj_by_factor |
                fpn.recall('C') |
                op.adj_by_factor |
                add_pni
        )

        class PNI(fpn.PipeNodeInput):
            def __init__(self, value):
                super().__init__()
                self.value = value

        pni = PNI(-89)
        post = expr[pni]

        # Manually calculate
        expected = 12
        expected += pni.value
        A = expected
        expected *= -8
        expected = (pni.value * 13) + (expected / 7)
        B = expected
        expected += op.DELTA
        expected += Operations.DELTA
        expected = max(min(expected, 0), 100)
        C = expected
        expected *= 130.2
        expected = max(min(expected, 100), 200)
        expected = op.factor * (pni.value - expected)
        D = expected
        expected = op.factor * (pni.value - C)
        expected += pni.value

        self.assertEqual(-4571.513, expected)
        self.assertEqual(-4571.513, post)

        self.assertEqual(A, pni._store['A'])
        self.assertEqual(B, pni._store['B'])
        self.assertEqual(C, pni._store['C'])
        self.assertEqual(D, pni._store['D'])

        self.assertEqual(pni.store_items(), dict(A=A, B=B, C=C, D=D).items())

    def test_core_decorator(self):
        # Modify the core_decorators
        found_f = set()

        PREFIX = 'TestUnit.test_core_decorator.<locals>.'

        def decorator(f):
            found_f.add(repr(f).replace(PREFIX, '').split()[1])
            return f

        pipe_node = functools.partial(fpn.pipe_node, core_decorator=decorator)
        pipe_node_factory = functools.partial(fpn.pipe_node_factory, core_decorator=decorator)
        classmethod_pipe_node = functools.partial(fpn.classmethod_pipe_node, core_decorator=decorator)
        staticmethod_pipe_node_factory = functools.partial(fpn.staticmethod_pipe_node_factory, core_decorator=decorator)

        @pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def store(pni, ret_val, label):
            pni.store(label, ret_val)
            return ret_val


        @pipe_node_factory(fpn.PN_INPUT)
        def recall(pni, label):
            return pni.recall(label)

        # ---------------------------------------------------------------------

        @pipe_node_factory(fpn.PREDECESSOR_RETURN)
        def multiply(prev, val):
            return prev * val

        @pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def add_pni(pni, prev):
            return prev + pni.value

        @pipe_node
        def init(**kwargs):
            return 12

        @pipe_node_factory
        def func(arg1, arg2, **kwargs):
            return (kwargs[fpn.PN_INPUT].value * arg1) + (kwargs[fpn.PREDECESSOR_RETURN] / arg2)

        class Operations:
            DELTA = 0.23

            def __init__(self, factor):
                self.factor = factor

            @classmethod_pipe_node
            def add_delta(cls, **kwargs):
                return cls.DELTA + kwargs[fpn.PREDECESSOR_RETURN]

            @staticmethod_pipe_node_factory(fpn.PREDECESSOR_RETURN)
            def bound_prev(prev, lower, upper):
                return max(min(prev, lower), upper)

            @pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
            def adj_by_factor(self, pni, prev):
                return self.factor * (pni.value - prev)

        op = Operations(23.717)

        expr = (init |
                add_pni |
                store('A') |
                multiply(-8) |
                func(13, 7) |
                store('B') |
                op.add_delta |
                Operations.add_delta |
                op.bound_prev(0, 100) |
                store('C') |
                multiply(130.2) |
                Operations.bound_prev(100, 200) |
                op.adj_by_factor |
                store('D') |
                recall('A') |
                op.adj_by_factor |
                recall('B') |
                op.adj_by_factor |
                recall('C') |
                op.adj_by_factor |
                add_pni
        )

        class PNI(fpn.PipeNodeInput):
            def __init__(self, value):
                super().__init__()
                self.value = value

        pni = PNI(-89)
        post = expr[pni]
        self.assertEqual(-4571.513, post)

        self.assertSetEqual({'store', 'recall', 'init', 'multiply', 'func',
                'Operations.add_delta', 'Operations.bound_prev',
                'Operations.adj_by_factor', 'add_pni'}, found_f)

    def test_repr_a(self):
        @fpn.pipe_node_factory()
        def factory(arg1, arg2, *, kwarg):
            pass

        self.assertEqual('<PN: factory(1,2,kwarg=3)>', repr(factory(1, 2, kwarg=3)))
        self.assertEqual('<PN: factory(1,2,kwarg=3) | factory(1,2,kwarg=3)>', repr(factory(1, 2, kwarg=3) | factory(1, 2, kwarg=3)))
        self.assertEqual('<PN: factory(1,2,kwarg=3)>', str(factory(1, 2, kwarg=3)))
        self.assertEqual('<PN: factory(1,2,kwarg=3) | factory(1,2,kwarg=3)>', str(factory(1, 2, kwarg=3) | factory(1, 2, kwarg=3)))

        @fpn.pipe_node()
        def pn1():
            pass

        @fpn.pipe_node()
        def pn2():
            pass

        @fpn.pipe_node_factory()
        def pn3(arg1, *arg_list, kwarg, **kwargs):
            pass

        args = (1, 2, 3)
        kwargs = dict(kwarg=4, a=5, b=6)

        self.assertEqual('<PN: pn1+pn2 | pn3(1,2,3,kwarg=4,a=5,b=6)>', repr((pn1 + pn2) | pn3(*args, **kwargs)))
        self.assertEqual('<PN: pn1 | pn2 | pn3(1,2,3,kwarg=4,a=5,b=6) | pn2 | pn1>', repr(pn1 | pn2 | pn3(*args, **kwargs) | pn2 | pn1))
        self.assertEqual('<PN: pn1+pn2 | pn3(1,2,3,kwarg=4,a=5,b=6)>', str((pn1 + pn2) | pn3(*args, **kwargs)))
        self.assertEqual('<PN: pn1 | pn2 | pn3(1,2,3,kwarg=4,a=5,b=6) | pn2 | pn1>', str(pn1 | pn2 | pn3(*args, **kwargs) | pn2 | pn1))

    def test_repr_b(self):
        @fpn.pipe_node()
        def a():
            return 1
        @fpn.pipe_node()
        def b():
            return 2
        @fpn.pipe_node()
        def c():
            return 3
        @fpn.pipe_node()
        def d():
            return 4
        @fpn.pipe_node()
        def e():
            return 5

        neg_a = -a
        abs_b = abs(b)
        inv_c = ~c

        self.assertEqual('<PN: -a>', repr(neg_a))
        self.assertEqual('<PN: abs(b)>', repr(abs_b))
        self.assertEqual('<PN: ~c>', repr(inv_c))

        add = a + b
        sub = a - b
        mul = a * b
        div = a / b

        self.assertEqual('<PN: a+b>', repr(add))
        self.assertEqual('<PN: a-b>', repr(sub))
        self.assertEqual('<PN: a*b>', repr(mul))
        self.assertEqual('<PN: a/b>', repr(div))

        add_mul_sub = add*sub
        sub_div_add = sub/add
        mul_add_div = mul+div
        div_sub_mul = div-mul
        add_pow_add = add**add

        self.assertEqual('<PN: (a+b)*(a-b)>', repr(add_mul_sub))
        self.assertEqual('<PN: (a-b)/(a+b)>', repr(sub_div_add))
        self.assertEqual('<PN: (a*b)+(a/b)>', repr(mul_add_div))
        self.assertEqual('<PN: (a/b)-(a*b)>', repr(div_sub_mul))
        self.assertEqual('<PN: (a+b)**(a+b)>', repr(add_pow_add))

        add_mul_sub_eq = add_mul_sub==c
        sub_div_add_gt = sub_div_add>d
        mul_add_div_lt = mul_add_div<e
        div_sub_mul_ge = div_sub_mul>=c
        add_pow_add_le = add_pow_add<=d
        add_pow_add_ne = add_pow_add!=e

        self.assertEqual('<PN: ((a+b)*(a-b))==c>', repr(add_mul_sub_eq))
        self.assertEqual('<PN: ((a-b)/(a+b))>d>', repr(sub_div_add_gt))
        self.assertEqual('<PN: ((a*b)+(a/b))<e>', repr(mul_add_div_lt))
        self.assertEqual('<PN: ((a/b)-(a*b))>=c>', repr(div_sub_mul_ge))
        self.assertEqual('<PN: ((a+b)**(a+b))<=d>', repr(add_pow_add_le))
        self.assertEqual('<PN: ((a+b)**(a+b))!=e>', repr(add_pow_add_ne))

        complex_a = (((a <= ~b) >= c) + d - abs(e) * a ** ((b / -c) != (d == e)))
        complex_b = ((a <= b) >= (-c + (d - (e * a) ** b) / (c != abs(d)) == ~e))
        complex_c = (((-a <= abs(b)) >= (c + d)) - (e * a) ** (~b / c) != (d == e))
        complex_d = (((a <= b) >= c) + (~d - ((abs(-e) * a) ** b) / (c != d) == e))
        complex_e = (abs(a) <= b >= c + ~d - e * a ** b / c != -d == e)

        self.assertEqual('<PN: (((a<=~b)>=c)+d)-(abs(e)*(a**((b/-c)!=(d==e))))>', repr(complex_a))
        self.assertEqual('<PN: (a<=b)>=((-c+((d-((e*a)**b))/(c!=abs(d))))==~e)>', repr(complex_b))
        self.assertEqual('<PN: (((-a<=abs(b))>=(c+d))-((e*a)**(~b/c)))!=(d==e)>', repr(complex_c))
        self.assertEqual('<PN: ((a<=b)>=c)+((~d-((((abs(-e))*a)**b)/(c!=d)))==e)>', repr(complex_d))
        self.assertEqual('<PN: -d==e>', repr(complex_e))

    def test_repr_c(self):
        lambda_func = lambda x: x+2
        fn = fpn.pipe_node(lambda_func)
        self.assertEqual('<PN: lambda_func = lambda x: x+2>', repr(fn))
        self.assertEqual(
                "<PN: repr(fpn.pipe_node(lambda x: x+2)),>",
                repr(fpn.pipe_node(lambda x: x+2)),
        ) # Isn't really an easy way to parse the lambda expression alone.

    def test_cannot_store_twice(self):
        @fpn.pipe_node()
        def pn():
            return 12

        pni = fpn.PipeNodeInput()

        with self.assertRaises(KeyError):
            (pn | fpn.store('a') | fpn.store('a'))[pni]

    def test_call(self):
        calls = []

        @fpn.pipe_node(fpn.PN_INPUT)
        def pn1(pni):
            self.assertEqual('pni', pni)
            calls.append('pn1')
            return 13

        @fpn.pipe_node(fpn.PN_INPUT)
        def pn2(pni):
            self.assertEqual('pni', pni)
            calls.append('pn2')
            return 14

        @fpn.pipe_node(fpn.PN_INPUT)
        def pn3(pni):
            self.assertEqual('pni', pni)
            calls.append('pn3')
            return 15

        call = fpn.call(pn1, pn2, pn3)
        post = call['pni']
        self.assertEqual(15, post)

        self.assertListEqual(['pn1', 'pn2', 'pn3'], calls)

    def test_invalid_operations(self):
        @fpn.pipe_node()
        def pn1():
            pass

        @fpn.pipe_node()
        def pn2():
            pass

        class MissingOp:
            pass

        with self.assertRaises(NotImplementedError):
            pn1 >> pn2

        with self.assertRaises(NotImplementedError):
            MissingOp() >> pn1

        with self.assertRaises(NotImplementedError):
            pn1 << pn2

        with self.assertRaises(NotImplementedError):
            MissingOp() << pn1

        with self.assertRaises(NotImplementedError):
            pn1.partial(1, 2, a=3, b=4)

    def test_pn_states(self):
        @fpn.pipe_node_factory()
        def factory(*args, **kwargs):
            pass

        @fpn.pipe_node()
        def pn():
            pass

        self.assertEqual(fpn.PipeNode.State.PROCESS, (pn | factory()).call_state)
        self.assertEqual(fpn.PipeNode.State.EXPRESSION, pn.call_state)
        self.assertEqual(fpn.PipeNode.State.FACTORY, factory.call_state)

    def test_predecessor_property(self):
        testable = {}

        @fpn.pipe_node()
        def init():
            pass

        @fpn.pipe_node(fpn.PREDECESSOR_PN)
        def pn1(prev_pn):
            testable['pn1'] = prev_pn

        @fpn.pipe_node(fpn.PREDECESSOR_PN)
        def pn2(prev_pn):
            testable['pn2'] = prev_pn

        expr = (init | pn1 | pn2)
        self.assertEqual({}, testable)
        self.assertIsNone(pn1.predecessor)
        self.assertIsNone(pn2.predecessor)

        expr[None]
        self.assertEqual({'pn1': pn1.predecessor, 'pn2': pn2.predecessor}, testable)

    def test_ror(self):
        class MissingOR:
            def __init__(self, state):
                self.state = state

            def __call__(self, *args, **kwargs):
                print(args, kwargs)
                return self.state

        obj = MissingOR(37)

        @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN, fpn.PREDECESSOR_PN)
        def pn(pni, prev, prev_pn):
            self.assertIs(obj, prev_pn)
            self.assertEqual(37, prev)
            self.assertEqual('pni', pni)
            return 343


        expr = (obj | pn)
        post = expr['pni']
        self.assertEqual(343, post)

    def test_invalid_factories(self):
        @fpn.pipe_node_factory
        def bad_factory_a(**kwargs):
            pass

        @fpn.pipe_node_factory()
        def bad_factory_b(pn_input):
            pass

        @fpn.pipe_node_factory()
        def good_factory(a, b, *, c, d=None):
            pass

        pn = fpn.pipe_node()(lambda :None)

        # Cannot use reserved kwargs!
        for kwarg in fpn.PIPE_NODE_KWARGS:
            with self.assertRaises(ValueError):
                bad_factory_a(**{kwarg: 'something'})

            with self.assertRaises(ValueError):
                bad_factory_b(**{kwarg: 'something'})

        # Cannot put a factory inside a pipeline!
        with self.assertRaises(ValueError):
            pn | good_factory

        # Calling a factory with an inadequate number of args/kwargs
        expr_should_fail = (pn | good_factory(1, d=14))
        with self.assertRaises(TypeError):
            expr_should_fail[None]

        # Seeding factory with nothing, and then passing in reserved kwargs
        with self.assertRaises(ValueError):
            bad_factory_a()(pn_input=True, predecessor_return=True)

    def test_invalid_pn_call(self):
        @fpn.pipe_node()
        def pn_a():
            pass

        @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def pn_b(pni, prev):
            pass

        with self.assertRaises(ValueError):
            pn_a(1, 2, 3)

        with self.assertRaises(ValueError):
            pn_b(1, 2, 3)

        with self.assertRaises(ValueError):
            pn_a(kwarg=13) # pylint: disable=unexpected-keyword-arg

        with self.assertRaises(ValueError):
            pn_b(kwarg=13) # pylint: disable=unexpected-keyword-arg

    def test_unwrapping(self):
        @fpn.pipe_node
        def pn1(*args, **kwargs):
            return 1

        @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
        def pn2(pni, prev):
            return pni + prev * 2

        @fpn.pipe_node_factory
        def pn3(a, *, b, c=None, **kwargs):
            return a+b-(c if c else 0)+kwargs[fpn.PREDECESSOR_RETURN]

        @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
        def pn4(prev, val):
            return prev**val

        def test_pn1(pn_or_expr):
            self.assertEqual(1, pn_or_expr.unwrap(1,2,3,4,5,6))
            self.assertEqual(1, pn_or_expr.unwrap(kwargs_accept_anything=True))
            self.assertIsInstance(pn_or_expr.unwrap, types.FunctionType)
            self.assertFalse(isinstance(pn_or_expr.unwrap, fpn.FunctionNode))

        test_pn1(pn1)
        test_pn1(pn2 | pn1)
        test_pn1(pn2 | pn4() | pn1)

        def test_pn2(pn_or_expr):
            with self.assertRaises(KeyError):
                pn_or_expr.unwrap(1, 2)

            with self.assertRaises(KeyError):
                pn_or_expr.unwrap(pni=1, prev=2)

            self.assertEqual(16, pn_or_expr.unwrap(pn_input=6, predecessor_return=5))
            self.assertIsInstance(pn_or_expr.unwrap, types.FunctionType)
            self.assertFalse(isinstance(pn_or_expr.unwrap, fpn.FunctionNode))

        test_pn2(pn2)
        test_pn2(pn1 | pn2)
        test_pn2(pn1 | pn3() | pn2)

        def test_pn3(pn_or_expr):
            with self.assertRaises(KeyError):
                pn_or_expr.unwrap(1, b=2, c=8) # Missing predecessor_return

            self.assertEqual(17.8, pn_or_expr.unwrap(1,b=-.2, c=-3,predecessor_return=14))
            self.assertIsInstance(pn_or_expr.unwrap, types.FunctionType)
            self.assertFalse(isinstance(pn_or_expr.unwrap, fpn.FunctionNode))

        test_pn3(pn3)
        test_pn3(pn1 | pn3())
        test_pn3(pn2 | pn4(1) | pn3())

        # pn4
        def test_pn4(pn_or_expr):
            with self.assertRaises(KeyError):
                pn_or_expr.unwrap(1) # Missing predecessor_return

            with self.assertRaises(KeyError):
                pn_or_expr.unwrap(prev=1) # Missing predecessor_return

            self.assertEqual(4.871658325766914, pn_or_expr.unwrap(0.6, predecessor_return=14))
            self.assertIsInstance(pn_or_expr.unwrap, types.FunctionType)
            self.assertFalse(isinstance(pn_or_expr.unwrap, fpn.FunctionNode))

        test_pn4(pn4)
        test_pn4(pn2 | pn4())
        test_pn4(pn1 | pn3() | pn4())


if __name__ == '__main__':
    unittest.main()
