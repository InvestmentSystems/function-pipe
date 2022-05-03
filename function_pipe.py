"""
function_pipe.py

Copyright 2012-2022 Research Affiliates

Authors: Christopher Ariza, Max Moroz, Charles Burkland

Common usage:
import function_pipe as fpn
"""
import enum
import functools
import inspect
import re
import sys
import types
import typing as tp


# -------------------------------------------------------------------------------
# FunctionNode and utilities

FN = tp.TypeVar("FN", bound="FunctionNode")
PN = tp.TypeVar("PN", bound="PipeNode")



def compose(*funcs):
    """
    Given a list of functions, execute them from right to left, passing
    the returned value of the right f to the left f. Store the reduced function in a FunctionNode
    """
    # call right first, then left of each pair; each reduction retruns a function
    reducer = functools.reduce(
        lambda f, g: lambda *args, **kaargs: f(g(*args, **kaargs)), funcs
    )
    # args are reversed to show execution from right to left
    return FunctionNode(reducer, doc_function=compose, doc_args=reversed(funcs))


def _wrap_unary(func: tp.Callable[[FN], FN]) -> tp.Callable[[FN], FN]:
    """
    Decorator for operator overloads. Given a higher order function that takes one args, wrap it in a FunctionNode function and provide documentation labels.
    """

    @functools.wraps(func)
    def unary(lhs: tp.Callable) -> FN:
        # wrapped function will prepare correct class, even if a constant
        cls = PipeNode if isinstance(lhs, PipeNode) else FunctionNode
        return cls(func(lhs), doc_function=func, doc_args=(lhs,))

    return unary


def _wrap_binary(func: tp.Callable[[FN, tp.Any], FN]) -> tp.Callable[[FN, tp.Any], FN]:
    """Decorator for operators. Given a higher order function that takes two args, wrap it in a FunctionNode function and provide documentation labels."""

    def binary(lhs: FN, rhs: tp.Any) -> FN:
        # wrapped function will prepare correct class, even if a constant
        cls = PipeNode if isinstance(lhs, PipeNode) else FunctionNode
        return cls(func(lhs, rhs), doc_function=func, doc_args=(lhs, rhs))

    return binary


_BINARY_OP_MAP = {
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__truediv__": "/",
    "__pow__": "**",
    "__eq__": "==",
    "__lt__": "<",
    "__le__": "<=",
    "__gt__": ">",
    "__ge__": ">=",
    "__ne__": "!=",
}


_UNARY_OP_MAP = {
    "__neg__": "-",
    "__invert__": "~",
    "__abs__": "abs",
}


def _contains_expression(repr_str: str) -> bool:
    """
    Checks whether or not a `repr_str` contains an expression. (Single unary expressions are excluded)
    """
    repr_str = re.sub(r"\s+", "", repr_str)
    repr_str = repr_str.replace("(", "")
    repr_str = repr_str.replace(")", "")

    symbols = re.findall(r"[\w']+", repr_str)
    non_symbols = re.findall(r"[^\w']+", repr_str)

    if len(non_symbols) == 0:
        return False

    if len(non_symbols) == 1 and len(symbols) == 1:
        return False

    return True


def _format_expression(f) -> str:
    """
    `f` could be either a single argument, or an expression of arguments. If it is the latter, wrap in parenthesis
    """
    repr_str = _repr(f)
    if _contains_expression(repr_str):
        return f"({repr_str})"
    return repr_str


def _repr(f) -> str:
    """Provide a string representation of the FN, recursively representing defined arguments."""

    def get_function_name(f) -> str:
        """Get a string representation of the callable, or its code if it is a lambda. In some cases, `f` may not be function, so just return a string."""
        if not isinstance(f, types.FunctionType) or not hasattr(f, "__name__"):
            return str(f)
        if f.__name__ == "<lambda>":
            # split on all white space, and rejoin with single space
            return " ".join(inspect.getsource(f).split())
        return f.__name__

    # find FunctionNode; using hasattr because of testing context issues
    if isinstance(f, FunctionNode):
        doc_f = get_function_name(f._doc_function)

        unary_op = _UNARY_OP_MAP.get(doc_f)
        binary_op = _BINARY_OP_MAP.get(doc_f)

        if unary_op:
            assert not f._doc_kwargs, "Unary FunctionNodes must not have doc_kwargs."
            assert (
                len(f._doc_args) == 1
            ), "Unary FunctionNodes must only have one doc_arg."

            if unary_op == "abs":
                arg = _repr(f._doc_args[0])
                return f"{unary_op}({arg})"

            arg = _format_expression(f._doc_args[0])
            return f"{unary_op}{arg}"

        if binary_op:
            assert not f._doc_kwargs, "Binary FunctionNodes must not have doc_kwargs."
            assert (
                len(f._doc_args) == 2
            ), "Binary FunctionNodes must only have two doc_args."

            left = _format_expression(f._doc_args[0])
            right = _format_expression(f._doc_args[1])

            return f"{left}{binary_op}{right}"

        if not f._doc_args and not f._doc_kwargs:
            return doc_f

        predecessor = ""
        sig_str = "("

        if f._doc_args:
            sig_str += ",".join((str(_repr(v)) for v in f._doc_args))

        if f._doc_kwargs:
            if f._doc_args:
                sig_str += ","

            for k, v in f._doc_kwargs.items():
                if k == PREDECESSOR_PN:
                    predecessor = _repr(v)
                else:
                    sig_str += (k + "=" + str(_repr(v))) + ","

        sig_str = sig_str.rstrip(",") + ")"

        if sig_str == "()":
            sig_str = doc_f
        else:
            sig_str = doc_f + sig_str

        if predecessor:
            sig_str = predecessor + " | " + sig_str
        return sig_str
    return get_function_name(f)


class FunctionNode:
    """
    A wrapper for a callable that can reside in an expression of numerous FunctionNodes, or be modified with unary or binary operators.
    """

    __slots__ = (
        "_function",
        "_doc_function",
        "_doc_args",
        "_doc_kwargs",
    )

    # ---------------------------------------------------------------------------
    def __init__(
            self: FN,
            function: tp.Any,
            *,
            doc_function: tp.Optional[tp.Callable] = None,
            doc_args: tp.Tuple[tp.Any, ...] = (),
            doc_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        ) -> None:
        """
        Args:

            - ``function``: a callable or value. If given a value, will create a function that simply returns that value.
            - ``doc_function``: the function to display in the repr; will be set to ``function`` if not provided
            - ``doc_args``: the positional arguments to display in the repr
            - ``doc_kwargs``: the keyword arguments to display in the repr

        """
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
    def unwrap(self: FN) -> tp.Callable:
        """
        The doc_function should be set to the core function being wrapped, no matter the level of wrapping.
        """
        # if the stored function is using _pipe_kwarg_bind, need to go lower
        doc_func = self
        while hasattr(doc_func, "_doc_function"):
            doc_func = getattr(doc_func, "_doc_function")
        return doc_func

    def __call__(self: FN, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """
        Call the wrapped function with args and kwargs.
        """
        return self._function(*args, **kwargs)

    def __str__(self: FN) -> str:
        return f"<FN: {_repr(self)}>"

    def __repr__(self: FN) -> str:
        return f"<FN: {_repr(self)}>"

    def partial(self: FN, *args: tp.Any, **kwargs: tp.Any) -> "FunctionNode":
        """
        Return a new FunctionNode with a partialed function with args and kwargs.
        """
        fn = FunctionNode(functools.partial(self._function, *args, **kwargs))

        for attr in self.__slots__:
            if not getattr(fn, attr):
                setattr(fn, attr, getattr(self, attr))

        return fn

    # -------------------------------------------------------------------------
    # Unary Operators

    @_wrap_unary
    def __neg__(self: FN) -> FN:
        """
        Return a new FunctionNode that when evaulated, will negate the result of ``self``
        """
        return lambda *args, **kwargs: self(*args, **kwargs) * -1

    @_wrap_unary
    def __invert__(self: FN) -> FN:
        """
        Return a new FunctionNode that when evaulated, will invert the result of ``self``

        NOTE:
            This is generally expected to be a Boolean inversion, such as ~ (not) applied to a Numpy, Pandas, or Static-Frame objects.
        """
        return lambda *args, **kwargs: self(*args, **kwargs).__invert__()

    @_wrap_unary
    def __abs__(self: FN) -> FN:
        """
        Return a new FunctionNode that when evaulated, will find the absolute value of the result of ``self``

        NOTE:
             Will use ``numpy.abs`` if available, as the most common usage is for Numpy, Pandas, or Static-Frame objects.
        """
        try:
            from numpy import abs as abs_func # Limit import overhead and dependencies
        except ImportError:
            abs_func = abs

        return lambda *args, **kwargs: abs_func(self(*args, **kwargs))

    # ---------------------------------------------------------------------------
    # all binary operators return a function; the _wrap_binary decorator then wraps this function in a FunctionNode definition and supplies appropriate doc args. Note both left and righ sides are wrapped in FNs to permit operations on constants

    @_wrap_binary
    def __add__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will add ``rhs`` to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) + self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __sub__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will subtract ``rhs`` to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) - self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __mul__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will multiply ``rhs`` by the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) * self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __truediv__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will divide ``rhs`` by the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) / self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __pow__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will divide raise the result of ``self`` by ``rhs``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) ** self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __radd__(self: FN, lhs) -> FN:
        """
        Return a new FunctionNode will add ``rhs`` to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(lhs)(
            *args, **kwargs
        ) + self.__class__(self)(*args, **kwargs)

    @_wrap_binary
    def __rsub__(self: FN, lhs) -> FN:
        """
        Return a new FunctionNode will subtract ``rhs`` to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(lhs)(
            *args, **kwargs
        ) - self.__class__(self)(*args, **kwargs)

    @_wrap_binary
    def __rmul__(self: FN, lhs) -> FN:
        """
        Return a new FunctionNode will multiply ``rhs`` by the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(lhs)(
                    """
        Return a new FunctionNode will test if ``rhs``' is greater than or equal to the result of ``self``
        """*args, **kwargs
        ) * self.__class__(self)(*args, **kwargs)

    @_wrap_binary
    def __rtruediv__(self: FN, lhs) -> FN:
        """
        Return a new FunctionNode will divide ``rhs`` by the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(lhs)(
            *args, **kwargs
        ) / self.__class__(self)(*args, **kwargs)

    # comparison operators, expected to return booleans
    @_wrap_binary
    def __eq__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' equals the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) == self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __lt__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is less than the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) < self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __le__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is less than or equal to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) <= self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __gt__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is greater than the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) > self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __ge__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is greater than or equal to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) >= self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __ne__(self: FN, rhs) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is not equal to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(
            *args, **kwargs
        ) != self.__class__(rhs)(*args, **kwargs)

    # ---------------------------------------------------------------------------
    # composition operators

    def __rshift__(self: FN, rhs: tp.Callable) -> FN:
        """
        Composes a new FunctionNode will call ``lhs`` first, and then feed its result into ``rhs``
        """
        return compose(rhs, self)

    def __rrshift__(self: FN, lhs: tp.Callable) -> FN:
        """
        Composes a new FunctionNode will call ``lhs`` first, and then feed its result into ``rhs``
        """
        return compose(self, lhs)

    def __lshift__(self: FN, rhs: tp.Callable) -> FN:
        """
        Composes a new FunctionNode will call ``rhs`` first, and then feed its result into ``lhs``
        """
        return compose(self, rhs)

    def __rlshift__(self: FN, lhs: tp.Callable) -> FN:
        """
        Composes a new FunctionNode will call ``rhs`` first, and then feed its result into ``lhs``
        """
        return compose(lhs, self)

    def __or__(self: FN, rhs: FN) -> FN:
        """
        Only implemented for PipeNode.
        """
        raise NotImplementedError()

    def __ror__(self: FN, lhs: FN) -> FN:
        """
        Only implemented for PipeNode.
        """
        raise NotImplementedError()


# -------------------------------------------------------------------------------
# PipeNode and utiltiies

# PipeNode kwargs
PREDECESSOR_RETURN = "predecessor_return"
PREDECESSOR_PN = "predecessor_pn"
PN_INPUT = "pn_input"
PN_INPUT_SET = frozenset((PN_INPUT,))
PREDECESSOR_PN_SET = frozenset((PREDECESSOR_PN,))
PIPE_NODE_KWARGS = frozenset((PREDECESSOR_RETURN, PREDECESSOR_PN, PN_INPUT))


class PipeNode(FunctionNode):
    """The multi-call structure of PipeNodes moves a FunctionNode between three states."""

    class State(enum.Enum):
        """The current state of the PipeNode"""

        FACTORY = "FACTORY"
        EXPRESSION = "EXPRESSION"
        PROCESS = "PROCESS"

    __slots__ = FunctionNode.__slots__ + (
        "_call_state",
        "_predecessor",
    )

    # ---------------------------------------------------------------------------
    def __init__(
        self,
        function,
        *,
        doc_function=None,
        doc_args=None,
        doc_kwargs=None,
        call_state=None,
        predecessor=None,
    ):
        super().__init__(
            function=function,
            doc_function=doc_function,
            doc_args=doc_args,
            doc_kwargs=doc_kwargs,
        )
        self._call_state: self.State = call_state
        self._predecessor = predecessor

    def __str__(self):
        if self._call_state is PipeNode.State.FACTORY:
            return f"<PNF: {_repr(self)}>"
        return f"<PN: {_repr(self)}>"

    def __repr__(self):
        if self._call_state is PipeNode.State.FACTORY:
            return f"<PNF: {_repr(self)}>"
        return f"<PN: {_repr(self)}>"

    def partial(self, *args, **kwargs):
        """PipeNode calling is dictated by the PipeNode protocol; partial-like behavior in expressions shold be achived with functions decorated with the  pipe_node_factory decorator."""
        raise NotImplementedError()

    # ---------------------------------------------------------------------------
    # pipe node properties

    @property
    def call_state(self) -> "State":
        """The current call state of the Node"""
        return self._call_state

    @property
    def predecessor(self):
        """The Node preceeding this Node in a pipeline. Can be None"""
        return self._predecessor

    # ---------------------------------------------------------------------------
    # composition operators

    def __rshift__(self, rhs):
        """Only implemented for FunctionNode."""
        raise NotImplementedError()

    def __rrshift__(self, lhs):
        """Only implemented for FunctionNode."""
        raise NotImplementedError()

    def __lshift__(self, rhs):
        """Only implemented for FunctionNode."""
        raise NotImplementedError()

    def __rlshift__(self, lhs):
        """Only implemented for FunctionNode."""
        raise NotImplementedError()

    def __or__(self, rhs):
        """Call RHS with LHS as an argument; left is passed as kwarg PREDECESSOR_PN. This calls the RHS immediately and does not return an FN unless prepared as a PipeNode"""
        return rhs(**{PREDECESSOR_PN: self})

    def __ror__(self, lhs):
        return self(**{PREDECESSOR_PN: lhs})

    # ---------------------------------------------------------------------------

    def __getitem__(self, pn_input):
        """
        Call self with some initial input `pn_input`. If None, will evaluate self with a default `PipeNodeInput` instance

        If desired initial input is literally `None`, use `.(**{PN_INPUT: None})` instead.
        """
        pn_input = pn_input if pn_input is not None else PipeNodeInput()
        return self(**{PN_INPUT: pn_input})

    def __call__(self, *args, **kwargs):
        """Call the wrapped function."""
        if self._call_state is PipeNode.State.FACTORY:
            return self._function(*args, **kwargs)
        if args or set(kwargs) - PIPE_NODE_KWARGS != set():
            raise ValueError(
                "Cannot call a PipeNode with args or non-pipeline kwargs! Please refer to the documentation for proper usage."
            )
        return self._function(**kwargs)


# -------------------------------------------------------------------------------
# decorator utilities


def _broadcast(*, factory_args, factory_kwargs, processing_args=(), processing_kwargs):
    """Factor args/kwargs are those given to pipe_node_factory at the expression level. Processing args/kwargs are those given as the initial input, and used to call all processing functions. After calling factor args with processing args, the result is used as core_callable args"""
    core_callable_args = [
        arg(*processing_args, **processing_kwargs) if isinstance(arg, PipeNode) else arg
        for arg in factory_args
    ]

    core_callable_kwargs = {
        kw: arg(*processing_args, **processing_kwargs)
        if isinstance(arg, PipeNode)
        else arg
        for kw, arg in factory_kwargs.items()
    }

    return core_callable_args, core_callable_kwargs


def core_logger(core_callable):
    """A decorator to provide output on the execution of each core callable call. Alternative decorators can be used to partial pipe_node_factory and pipe_node."""

    @functools.wraps(core_callable)
    def wrapped(*args, **kwargs):
        print("|", str(core_callable), file=sys.stderr)
        return core_callable(*args, **kwargs)

    return wrapped


def _has_key_positions(*key_positions):
    return not bool(len(key_positions) == 1 and callable(key_positions[0]))


def _is_unbound_self_method(core_callable):
    """Inspects a given callable to determine if it's both unbound, and the first argument in it's signature is `self`"""
    if isinstance(core_callable, types.MethodType):
        return False

    if isinstance(core_callable, (staticmethod, classmethod)):
        return False

    if isinstance(core_callable, functools.partial):
        return False

    argspec = inspect.getfullargspec(core_callable)
    return bool(argspec.args and argspec.args[0] == "self")


class PipeNodeDescriptor:  # pylint: disable=too-few-public-methods
    """
    Wraps up `pipe_node`/`pipe_node_factory` behavior in a descriptor, where it will bind instance and owner to the core_callable, and then pass it along the pipeline
    """

    __slots__ = ("core_callable", "core_handler", "key_positions")

    def __init__(self, core_callable, core_handler, key_positions=()):
        self.core_callable = core_callable
        self.core_handler = core_handler
        self.key_positions = key_positions

    def __get__(self, instance, owner):
        core_callable = self.core_callable.__get__(instance, owner)
        if self.key_positions:
            core_callable = _pipe_kwarg_bind(*self.key_positions)(core_callable)
        return self.core_handler(core_callable)


def _handle_descriptors_and_key_positions(*key_positions, core_handler):
    has_key_positions = _has_key_positions(*key_positions)

    # See if decorator was given no arguments, and received the core_callable directly.
    if not has_key_positions:
        final_callable = key_positions[0]

        if _is_unbound_self_method(final_callable):
            return PipeNodeDescriptor(final_callable, core_handler)

        return core_handler(final_callable)

    def decorator_wrapper(core_callable):
        if _is_unbound_self_method(core_callable):
            return PipeNodeDescriptor(core_callable, core_handler, key_positions)

        final_callable = _pipe_kwarg_bind(*key_positions)(core_callable)
        return core_handler(final_callable)

    return decorator_wrapper


def _pipe_kwarg_bind(*key_positions):
    """
    Binds n specific PN labels wrapped up in **kwargs to the first n positional arguments of the core callable
    """

    def decorator(core_callable):
        @functools.wraps(core_callable)
        def wrapped(*args, **kwargs):
            target_args = [kwargs.pop(key) for key in key_positions]
            target_kwargs = {
                k: v for k, v in kwargs.items() if k not in PIPE_NODE_KWARGS
            }
            return core_callable(*target_args, *args, **target_kwargs)

        return wrapped

    return decorator


def _descriptor_factory(*key_positions, decorator, core_decorator):
    has_key_positions = _has_key_positions(*key_positions)

    class Descriptor:  # pylint: disable=too-few-public-methods
        def __init__(self, func):
            self._func = func

        def __get__(self, instance, owner):
            # Prefer this to partialing for prettier func reprs
            @functools.wraps(self._func)
            def func(*args, **kwargs):
                return self._func(owner, *args, **kwargs)

            if has_key_positions:
                func = _pipe_kwarg_bind(*key_positions)(func)
            return decorator(func, core_decorator=core_decorator)

    if not has_key_positions:
        return Descriptor(key_positions[0])

    return Descriptor


# -------------------------------------------------------------------------------
# decorators


def pipe_node_factory(*key_positions, core_decorator=core_logger):
    """
    Returns a factory, that when given some args/kwargs, will return a PipeNode.

    Calling with arguments results in those arguments being positionally bound to the first arguments in the core_callable

    **Example:**

    >>> @fpn.pipe_node_factory
    >>> def func_a(arg1, arg2, **kwargs):
    >>>     pass

    >>> func_a(1, 2) # This is now a PipeNode ready to be bound in a pipeline

    >>> @fpn.pipe_node_factory(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
    >>> def func_b(pni, prev_val, arg1, arg2):
    >>>     # pni will be given the fpn.PN_INPUT from the pipeline
    >>>     # prev will be given the fpn.PREDECESSOR_RETURN from the pipeline
    >>>     pass

    >>> func_b(1, 2) # This is now a PipeNode ready to be bound in a pipeline
    """

    def build_factory(core_callable):
        decorated_core_callable = core_decorator(core_callable)

        def factory_f(*f_args, **f_kwargs):
            """This is the function returned by the decorator, used to create the PipeNode that resides in expressions after being called with arguments.

            f_args and f_kwargs are passed to the core_callable; if f_args or f_kwargs are PipeNode instances, they will be called with the processing args and kwargs (including PN_INPUT), either from process_f or (if innermost) from expression args.
            """
            if set(f_kwargs).intersection(PIPE_NODE_KWARGS) != set():
                raise ValueError(
                    f"Either you put a factory in a pipeline (i.e. not a pipe node), or your factory was given a reserved pipeline kwarg {tuple(PIPE_NODE_KWARGS)}."
                )

            def expression_f(**e_kwargs):
                """This is the PipeNode that resides in expressions prior to `|` operator evalation.
                When called with `|`, the predecessor is passed is in e_kwargs as PREDECESSOR_PN. In this usage the e_args will always be empty.

                When in the innermost position, expression_f is never called with `|` but with the PN_INPUT; this sitation is identified and the core_callable is called immediately.

                e_args will only be used as an innermost call.
                """
                kwargset = set(e_kwargs)
                if kwargset not in (PN_INPUT_SET, PREDECESSOR_PN_SET):
                    raise ValueError(
                        f"Expected to be called with either {PN_INPUT} or {PREDECESSOR_PN} only, not: {kwargset}"
                    )

                # identify innermost condition as when the expression level kwargs consists only of PN_INPUT
                if kwargset == PN_INPUT_SET:
                    # as this is innermost, processing args (i.e., PipeNodeInput) are given here at the expression level (as no Pipe operator has been used to call the innermost)
                    core_callable_args, core_callable_kwargs = _broadcast(
                        factory_args=f_args,
                        factory_kwargs=f_kwargs,
                        processing_kwargs=e_kwargs,
                    )  # not p_kwargs

                    # pack PipeNode protocol kwargs; when used as innermost, a core_callable can only expect to have a PN_INPUT
                    core_callable_kwargs[PN_INPUT] = e_kwargs[PN_INPUT]

                    return decorated_core_callable(
                        *core_callable_args, **core_callable_kwargs
                    )

                predecessor_pn = e_kwargs[PREDECESSOR_PN]

                def process_f(*p_args, **p_kwargs):
                    # call the predecssor PipeNode (here a process_f) with these processing args; these are always the args given as the initial input to the innermost function, generally a PipeNodeInput
                    predecessor_return = predecessor_pn(*p_args, **p_kwargs)

                    core_callable_args, core_callable_kwargs = _broadcast(
                        factory_args=f_args,
                        factory_kwargs=f_kwargs,
                        processing_args=p_args,
                        processing_kwargs=p_kwargs,
                    )

                    # pack PipeNode protocol kwargs
                    core_callable_kwargs[PN_INPUT] = p_kwargs[PN_INPUT]
                    core_callable_kwargs[PREDECESSOR_PN] = predecessor_pn
                    core_callable_kwargs[PREDECESSOR_RETURN] = predecessor_return

                    return decorated_core_callable(
                        *core_callable_args, **core_callable_kwargs
                    )

                assert (
                    set(e_kwargs).intersection(set(f_kwargs)) == set()
                )  # This is in impossible state

                # we must return a PipeNode here, as this is the final thing returned and might be passed on to another series func
                return PipeNode(
                    process_f,
                    doc_function=core_callable,
                    doc_args=f_args,
                    doc_kwargs={**e_kwargs, **f_kwargs},
                    call_state=PipeNode.State.PROCESS,
                    predecessor=predecessor_pn,
                )

            return PipeNode(
                expression_f,
                doc_function=core_callable,
                doc_args=f_args,
                doc_kwargs=f_kwargs,
                call_state=PipeNode.State.EXPRESSION,
            )

        # return a function node so as to make doc_function available in test
        return PipeNode(
            factory_f, doc_function=core_callable, call_state=PipeNode.State.FACTORY
        )

    return _handle_descriptors_and_key_positions(
        *key_positions, core_handler=build_factory
    )


def pipe_node(*key_positions, core_decorator=core_logger):
    """Decorate a function that takes no expression-level args."""

    def create_factory_and_call_once(core_callable):
        # Create a factory and call it once with no args to get an expresion-level function
        pnf = pipe_node_factory(core_callable, core_decorator=core_decorator)

        if not callable(pnf):
            raise ValueError(f"{core_callable.__qualname__} requires an instance")

        return pipe_node_factory(core_callable, core_decorator=core_decorator)()

    return _handle_descriptors_and_key_positions(
        *key_positions, core_handler=create_factory_and_call_once
    )


def classmethod_pipe_node_factory(*key_positions, core_decorator=core_logger):
    return _descriptor_factory(
        *key_positions, decorator=pipe_node_factory, core_decorator=core_decorator
    )


def classmethod_pipe_node(*key_positions, core_decorator=core_logger):
    return _descriptor_factory(
        *key_positions, decorator=pipe_node, core_decorator=core_decorator
    )


staticmethod_pipe_node_factory = pipe_node_factory
staticmethod_pipe_node = pipe_node


# -------------------------------------------------------------------------------
# PipeNodeInput


class PipeNodeInput:
    """PipeNode input to support store and recall; subclassable to expose other attributes and parameters."""

    def __init__(self):
        self._store = {}

    def store(self, key, value):
        if key in self._store:
            raise KeyError("cannot store the same key", key)
        self._store[key] = value

    def recall(self, key):
        return self._store[key]

    def store_items(self):
        return self._store.items()


# -------------------------------------------------------------------------------
# utility PipeNodes


@pipe_node_factory(PN_INPUT, PREDECESSOR_RETURN)
def store(pni, ret_val, label):
    pni.store(label, ret_val)
    return ret_val


@pipe_node_factory(PN_INPUT)
def recall(pni, label):
    return pni.recall(label)


@pipe_node_factory()
def call(*pns):
    """Call the PipeNode arguments with the PipeNodeInput as necessary (which happens in the broadcast routine in handling *args)"""
    return pns[-1]  # the last result is returned
