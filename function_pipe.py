'''
function_pipe.py

Copyright 2012-2017 Research Affiliates

Authors: Christopher Ariza, Max Moroz

Common usage:
import function_pipe as fpn
'''

import functools
import inspect
import collections
import types
import sys


#-------------------------------------------------------------------------------
# FunctionNode utilities

def compose(*funcs):
    '''
    Given a list of functions, execute them from right to left, passing
    the returned value of the right f to the left f. Store the reduced function in a FunctionNode
    '''
    # call right first, then left of each pair; each reduction retruns a function
    reducer = functools.reduce(lambda f, g:
            lambda *args, **kaargs: f(g(*args, **kaargs)), funcs)
    # args are reversed to show execution from right to left
    return FunctionNode(reducer, doc_function=compose, doc_args=reversed(funcs))

def _wrap_unary(func):
    '''Decorator for operator overloads. Given a higher order function that takes one args, wrap it in a FunctionNode function and provide documentation labels.
    '''
    def unary(lhs):
        # wrapped function will prepare correct class, even if a constant
        cls = PipeNode if isinstance(lhs, PipeNode) else FunctionNode
        return cls(func(lhs),
            doc_function=func,
            doc_args=(lhs,)
            )
    return unary

def _wrap_binary(func):
    '''Decorator for operators. Given a higher order function that takes two args, wrap it in a FunctionNode function and provide documentation labels.
    '''
    def binary(lhs, rhs):
        # wrapped function will prepare correct class, even if a constant
        cls = PipeNode if isinstance(lhs, PipeNode) else FunctionNode
        return cls(func(lhs, rhs),
            doc_function=func,
            doc_args=(lhs, rhs)
            )
    return binary


def _repr(f, doc_args=True):
    '''Provide a string representation of the FN, recursively representing defined arguments.
    '''
    def get_function_name(f):
        '''Get a string representation of the callable, or its code if it is a lambda. In some cases, `f` may not be function, so just return a string.
        '''
        f_type = type(f)
        if f_type is not types.FunctionType or not hasattr(f, '__name__'):
            # functool partial types do not have __name__ attrs, and are not FunctionTypes
            return str(f)
        if f.__name__ == '<lambda>':
            # split on all white space, and rejoin with single space
            return ' '.join(inspect.getsource(f).split())
        return f.__name__

    # find FunctionNode; using hasattr because of testing context issues
    if hasattr(f, '_doc_function'):
        if f._doc_function:
            doc_f = get_function_name(f._doc_function)
            if doc_args:
                args = kwargs = ''
                if f._doc_args:
                    args = (str(_repr(v)) for v in f._doc_args)
                if f._doc_kwargs:
                    kwargs = (k + '=' + str(_repr(f)) for k, v
                        in f._doc_kwargs.items())
                if not args and not kwargs:
                    return doc_f
                return doc_f + '(' + ','.join(args) + ','.join(kwargs) + ')'
            return doc_f
        else: # we don't know its structure, use _function
            return get_function_name(f._function)
    return get_function_name(f)



class FunctionNode:
    '''A wrapper for a callable that can reside in an expression of numerous FunctionNodes, or be modified with unary or binary operators.
    '''
    __slots__ = (
            '_function',
            '_doc_function',
            '_doc_args',
            '_doc_kwargs',
            )

    #---------------------------------------------------------------------------
    def __init__(self,
            function,
            *,
            doc_function=None,
            doc_args=None,
            doc_kwargs=None,
            call_state=None,
            predecessor=None
            ):
        '''
        Args:
            function: a callable
            doc_function: the function to display; will be set to `function` if nor provided
        '''
        # if a function node, re-wrap
        if isinstance(function, FunctionNode):
            for attr in self.__slots__:
                setattr(self, attr, getattr(function, attr))
        else:
            if callable(function):
                self._function = function
            else:
                # if not a callable, we upgrade a constant, non function value to be a function that returns that value
                self._function = lambda *args, **kwargs: function

            # if not supplied, doc_function is set to function
            self._doc_function = doc_function if doc_function else self._function
            self._doc_args = doc_args
            self._doc_kwargs = doc_kwargs

    @property
    def unwrap(self):
        '''The doc_function should be set to the core function being wrapped, no matter the level of wrapping.
        '''
        # if the stored function is using pipe_kwarg_bind, need to go lower
        doc_func = self
        while hasattr(doc_func, '_doc_function'):
            doc_func = getattr(doc_func, '_doc_function')
        return doc_func

    def __call__(self, *args, **kwargs):
        '''Call the wrapped function.
        '''
        return self._function(*args, **kwargs)

    def __str__(self):
        return '<FN: {}>'.format(_repr(self))

    __repr__ = __str__

    #__name__ = '<FN: {}>'.format(_repr(self, doc_args=False))
    #__name__ = __str__

    def partial(self, *args, **kwargs):
        '''Return a new FunctionNode with a partialed function with args and kwargs'
        '''
        fn = FunctionNode(functools.partial(self._function, *args, **kwargs))
        for attr in self.__slots__:
            if not getattr(fn, attr):
                setattr(fn, attr, getattr(self, attr))
        return fn

    #---------------------------------------------------------------------------
    # all unary operators return a function; the _wrap_unary decorator then wraps this function in a FunctionNode

    @_wrap_unary
    def __neg__(self):
        return lambda *args, **kwargs: self(*args, **kwargs) * -1

    @_wrap_unary
    def __invert__(self):
        '''This is generally expected to be a Boolean inversion, such as ~ (not) applied to a numpy array or pd.Series.
        '''
        return lambda *args, **kwargs: self(*args, **kwargs).__invert__()

    @_wrap_unary
    def __abs__(self):
        '''Absolute value; most common usage us on Numpy or Pandas objects, and thus here we np.abs.
        '''
        import numpy as np
        return lambda *args, **kwargs: np.abs(self(*args, **kwargs))

    #---------------------------------------------------------------------------
    # all binary operators return a function; the _wrap_binary decorator then wraps this function in a FunctionNode definition and supplies appropriate doc args. Note both left and righ sides are wrapped in FNs to permit operations on constants

    @_wrap_binary
    def __add__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) +
                lhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __sub__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) -
                lhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __mul__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) *
                lhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __truediv__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) /
                lhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __pow__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) **
                lhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __radd__(rhs, lhs):
        return (lambda *args, **kwargs:
                rhs.__class__(lhs)(*args, **kwargs) +
                rhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __rsub__(rhs, lhs):
        return (lambda *args, **kwargs:
                rhs.__class__(lhs)(*args, **kwargs) -
                rhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __rmul__(rhs, lhs):
        return (lambda *args, **kwargs:
                rhs.__class__(lhs)(*args, **kwargs) *
                rhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __rtruediv__(rhs, lhs):
        return (lambda *args, **kwargs:
                rhs.__class__(lhs)(*args, **kwargs) /
                rhs.__class__(rhs)(*args, **kwargs))

    # comparison operators, expected to return booleans
    @_wrap_binary
    def __eq__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) ==
                lhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __lt__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) <
                lhs.__class__(rhs)(*args, **kwargs))
    @_wrap_binary
    def __le__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) <=
                lhs.__class__(rhs)(*args, **kwargs))
    @_wrap_binary
    def __gt__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) >
                lhs.__class__(rhs)(*args, **kwargs))
    @_wrap_binary
    def __ge__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) >=
                lhs.__class__(rhs)(*args, **kwargs))

    @_wrap_binary
    def __ne__(lhs, rhs):
        return (lambda *args, **kwargs:
                lhs.__class__(lhs)(*args, **kwargs) !=
                lhs.__class__(rhs)(*args, **kwargs))

    #---------------------------------------------------------------------------
    # composition operators

    def __rshift__(lhs, rhs):
        '''Composition; return a function that will call LHS first, then RHS
        '''
        return compose(rhs, lhs)

    def __rrshift__(rhs, lhs):
        '''Composition; return a function that will call LHS first, then RHS
        '''
        return compose(rhs, lhs)

    def __lshift__(lhs, rhs):
        '''Composition; return a function that will call RHS first, then LHS
        '''
        return compose(lhs, rhs)

    def __llshift__(rhs, lhs):
        '''Composition; return a function that will call RHS first, then LHS
        '''
        return compose(lhs, rhs)

    def __or__(lhs, rhs):
        '''Only implemented for PipeNode.
        '''
        raise NotImplementedError

    def __ror__(rhs, lhs):
        '''Only implemented for PipeNode.
        '''
        raise NotImplementedError


#-------------------------------------------------------------------------------
# PipeNode and utiltiies

# PipeNode kwargs
PREDECESSOR_RETURN = 'predecessor_return'
PREDECESSOR_PN = 'predecessor_pn'
PN_INPUT = 'pn_input'
PN_INPUT_SET = {PN_INPUT}
PIPE_NODE_KWARGS = {PREDECESSOR_RETURN, PREDECESSOR_PN, PN_INPUT}


class PipeNode(FunctionNode):
    '''The multi-call structure of PipeNodes moves a FunctionNode between three states.
    '''

    # states
    FACTORY = 'FACTORY'
    EXPRESSION = 'EXPRESSION'
    PROCESS = 'PROCESS'

    __slots__ = FunctionNode.__slots__ + (
            '_call_state',
            '_predecessor'
            )

    #---------------------------------------------------------------------------
    def __init__(self,
                function,
                *,
                doc_function=None,
                doc_args=None,
                doc_kwargs=None,
                call_state=None,
                predecessor=None
                ):
        super().__init__(function=function,
                doc_function=doc_function,
                doc_args=doc_args,
                doc_kwargs=doc_kwargs
                )
        self._call_state = call_state
        self._predecessor = predecessor

    def __str__(self):
        return '<PN: {}>'.format(_repr(self))

    def partial(*args, **kwargs):
        '''PipeNode calling is dictated by the PipeNode protocol; partial-like behavior in expressions shold be achived with functions decorated with the  pipe_node_factory decorator.
        '''
        raise NotImplementedError

    #---------------------------------------------------------------------------
    # pipe node properties

    @property
    def call_state(self):
        return self._call_state

    @property
    def predecessor(self):
        return self._predecessor

    #---------------------------------------------------------------------------
    # composition operators

    def __rshift__(lhs, rhs):
        '''Only implemented for FunctionNode.
        '''
        raise NotImplementedError

    def __rrshift__(rhs, lhs):
        '''Only implemented for FunctionNode.
        '''
        raise NotImplementedError

    def __lshift__(lhs, rhs):
        '''Only implemented for FunctionNode.
        '''
        raise NotImplementedError

    def __llshift__(rhs, lhs):
        '''Only implemented for FunctionNode.
        '''
        raise NotImplementedError

    def __or__(lhs, rhs):
        '''Call RHS with LHS as an argument; left is passed as kwarg PREDECESSOR_PN. This calls the RHS immediately and does not return an FN unless prepared as a PipeNode
        '''
        return rhs(**{PREDECESSOR_PN:lhs})

    def __ror__(rhs, lhs):
        return rhs(**{PREDECESSOR_PN:lhs})


    #---------------------------------------------------------------------------
    def __getitem__(self, pn_input):
        '''Call self with the passed PipeNodeInput.
        '''
        pni = pn_input if pn_input else PipeNodeInput()
        return self(**{PN_INPUT:pni})


#-------------------------------------------------------------------------------
# decorator utilities

def _broadcast(factory_args,
        factory_kwargs,
        processing_args,
        processing_kwargs):
    '''Factor args/kwargs are those given to pipe_node_factory at the expression level. Processing args/kwargs are those given as the initial input, and used to call all processing functions. After calling factor args with processing args, the result is used as core_callable args
    '''
    core_callable_args = [arg(*processing_args, **processing_kwargs)
            if isinstance(arg, PipeNode) else arg
            for arg in factory_args]

    core_callable_kwargs = {kw: arg(*processing_args, **processing_kwargs)
            if isinstance(arg, PipeNode) else arg
            for kw, arg in factory_kwargs.items()}

    return core_callable_args, core_callable_kwargs


def core_logger(core_callable):
    '''A decorator to provide output on the execution of each core callable call. Alternative decorators can be used to partial pipe_node_factory and pipe_node.
    '''
    def wrapped(*args, **kwargs):
        prefix = '|'
        print('|', str(core_callable), file=sys.stderr)
        post = core_callable(*args, **kwargs)
        return post
    return wrapped

#-------------------------------------------------------------------------------
# decorators


def pipe_kwarg_bind(*key_positions):
    '''Using FN labels as arguments, define the what positional arguments of the wrapped function will receive from the common FN kwargs.
    '''
    def decorator(f):
        def wrapped(*args, **kwargs):
            # extract args from kwargs based on order of key_positions
            target_args = []
            for pos, k in enumerate(key_positions):
                target_args.append(kwargs.pop(k))
            target_kwargs = {k:v for k, v in kwargs.items()
                    if k not in PIPE_NODE_KWARGS}
            return f(*target_args, *args, **target_kwargs)
        return PipeNode(wrapped, doc_function=f)
    return decorator


def pipe_node_factory(core_callable,
        core_decorator=core_logger):
    '''This is a decorator.

    Upgrade keyword only arguments from a function that needs expression level args.
    '''
    decorated_core_callable = core_decorator(core_callable)

    def factory_f(*f_args, **f_kwargs):
        '''This is the function returned by the decorator, used to create the FunctionNode that resides in expressions after being called with arguments.

        f_args and f_kwargs are passed to the core_callable; if f_args or f_kwargs are FunctionNode instances, they will be called with the processing args and kwargs (including PN_INPUT), either from process_f or (if innermost) from expression args.
        '''
        def expression_f(*e_args, **e_kwargs):
            '''This is the FunctionNode that resides in expressions prior to `|` operator evalation. When called with `|`, the predecessor is passed is in e_kwargs as PREDECESSOR_PN. In this usage the e_args will always be empty.

            When in the innermost position, expression_f is never called with `|` but with the PN_INPUT; this sitation is identified and the core_callable is called immediately.

            e_args will only be used as an innermost call.
            '''
            # identify innermost condition as when the expression level kwargs consists only of PN_INPUT
            if set(e_kwargs.keys()) == PN_INPUT_SET:
                # as this is innermost, processing args (i.e., PipeNodeInput) are given here at the expression level (as no Pipe operator has been used to call the innermost)
                core_callable_args, core_callable_kwargs = _broadcast(
                        factory_args=f_args,
                        factory_kwargs=f_kwargs,
                        processing_args=e_args, # not p_args
                        processing_kwargs=e_kwargs) # not p_kwargs

                # pack PipeNode protocol kwargs; when used as innermost, a core_callable can only expect to have a PN_INPUT
                core_callable_kwargs[PN_INPUT] = e_kwargs[PN_INPUT]

                return decorated_core_callable(*core_callable_args,
                        **core_callable_kwargs)

            predecessor_pn = e_kwargs.get(PREDECESSOR_PN)

            def process_f(*p_args, **p_kwargs):
                # call the predecssor PipeNode (here a process_f) with these processing args; these are always the args given as the initial input to the innermost function, generally a PipeNodeInput
                predecessor_return = predecessor_pn(*p_args, **p_kwargs)

                core_callable_args, core_callable_kwargs = _broadcast(
                        factory_args=f_args,
                        factory_kwargs=f_kwargs,
                        processing_args=p_args,
                        processing_kwargs=p_kwargs)

                # pack PipeNode protocol kwargs
                core_callable_kwargs[PN_INPUT] = p_kwargs[PN_INPUT]
                core_callable_kwargs[PREDECESSOR_PN] = predecessor_pn
                core_callable_kwargs[PREDECESSOR_RETURN] = predecessor_return

                return decorated_core_callable(*core_callable_args,
                        **core_callable_kwargs)

            # we must return a PipeNode here, as this is the final thing returned and might be passed on to another series func
            return PipeNode(process_f,
                    doc_function=core_callable,
                    #doc_args=e_args,
                    #doc_kwargs=e_kwargs, # TODO: does not work
                    call_state=PipeNode.PROCESS,
                    predecessor=predecessor_pn)
        return PipeNode(expression_f,
                doc_function=core_callable,
                doc_args=f_args,
                doc_kwargs=f_kwargs,
                call_state=PipeNode.EXPRESSION)
    # return a function node so as to make doc_function available in test
    return PipeNode(factory_f,
            doc_function=core_callable,
            call_state=PipeNode.FACTORY)


def pipe_node(core_callable, core_decorator=core_logger):
    '''Decorate a function that takes no expression-level args.
    '''
    # create a factory and call it once with no args to get an expresion-level function
    return pipe_node_factory(core_callable,
            core_decorator=core_decorator)()


#-------------------------------------------------------------------------------
class PipeNodeInput:
    '''PipeNode input to support store and recall; subclassable to expose other attributes and parameters.
    '''

    def __init__(self):
        self._store = collections.OrderedDict()

    def store(self, key, value):
        if key in self._store:
            raise KeyError('cannot store the same key', key)
        self._store[key] = value

    def recall(self, key):
        return self._store[key]

    def store_items(self):
        return self._store.items()


#-------------------------------------------------------------------------------
# utility PipeNodes

@pipe_node_factory
def store(label, **kwargs):
    kwargs[PN_INPUT].store(label, kwargs[PREDECESSOR_RETURN])
    return kwargs[PREDECESSOR_RETURN]

@pipe_node_factory
def recall(label, **kwargs):
    return kwargs[PN_INPUT].recall(label)

@pipe_node_factory
def call(*args, **kwargs):
    '''Call the PipeNode arguments with the PipeNodeInput as necessary (which happens in the broadcast routine in handling *args)
    '''
    return args[-1] # the last result is returned

