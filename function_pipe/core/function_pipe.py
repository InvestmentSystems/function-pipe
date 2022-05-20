# pylint: disable=C0328
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
PNI = tp.TypeVar("PNI", bound="PipeNodeInput")
PipeNodeDescriptorT = tp.TypeVar("PipeNodeDescriptorT", bound="PipeNodeDescriptor")
KeyPostion = tp.Union[tp.Callable, str]
HandlerT = tp.Callable[[tp.Any], tp.Callable]


def compose(*funcs: tp.Callable) -> FN:
    """
    Given a list of functions, execute them from right to left, passing
    the returned value of the right f to the left f. Store the reduced function in a FunctionNode
    """
    # call right first, then left of each pair; each reduction retruns a function
    reducer = functools.reduce(
        lambda f, g: lambda *args, **kaargs: f(g(*args, **kaargs)), funcs
    )
    # args are reversed to show execution from right to left
    return FunctionNode(  # type: ignore
        reducer,
        doc_function=compose,
        doc_args=tuple(reversed(funcs)),
    )


def _wrap_unary(func: tp.Callable[[FN], FN]) -> tp.Callable[[FN], FN]:
    """
    Decorator for operator overloads. Given a higher order function that takes one args, wrap it in a FunctionNode function and provide documentation labels.
    """

    @functools.wraps(func)
    def unary(lhs: FN) -> FN:
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


def _from_this_module(tb: types.TracebackType) -> bool:
    return tb.tb_frame.f_code.co_filename == __file__


def _exception_with_cleaned_tb(original_exception: Exception) -> Exception:
    """
    Return back a new exception, where traceback is pointing to the first frame with code originating outside this module.
    """
    tb = original_exception.__traceback__
    assert tb is not None
    tb_next = None

    assert _from_this_module(tb)  # Sanity

    while True:
        tb_next = tb.tb_next

        # If `tb_next` is None, it means everything from the stack came from this module.
        # Use the original exception
        if tb_next is None:
            return original_exception

        # If `tb_next` originates from outside the module, it means we are done looking
        if not _from_this_module(tb_next):
            break

        # We are still observing frames from inside the module; keep looking
        tb = tb_next

    return original_exception.__class__(*original_exception.args).with_traceback(
        tb_next
    )


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


def _format_expression(f: tp.Any) -> str:
    """
    `f` could be either a single argument, or an expression of arguments. If it is the latter, wrap in parenthesis
    """
    repr_str = pretty_repr(f)
    if _contains_expression(repr_str):
        return f"({repr_str})"
    return repr_str


def pretty_repr(f: tp.Any) -> str:
    """
    Provide a pretty string representation of a FN, PN, or anything.
    If the object is a FN or PN, it will recursively represent any nested FNs/PNs.
    """

    def get_function_name(f: tp.Any) -> str:
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
                arg = pretty_repr(f._doc_args[0])
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
            sig_str += ",".join((str(pretty_repr(v)) for v in f._doc_args))

        if f._doc_kwargs:
            if f._doc_args:
                sig_str += ","

            for k, v in f._doc_kwargs.items():
                if k == PREDECESSOR_PN:
                    predecessor = pretty_repr(v)
                else:
                    sig_str += (k + "=" + str(pretty_repr(v))) + ","

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
        try:
            return self._function(*args, **kwargs)
        except Exception as e:
            raise _exception_with_cleaned_tb(e) from None

    def __str__(self: FN) -> str:
        return f"<FN: {pretty_repr(self)}>"

    def __repr__(self: FN) -> str:
        return f"<FN: {pretty_repr(self)}>"

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
        return lambda *args, **kwargs: self(*args, **kwargs) * -1  # type: ignore

    @_wrap_unary
    def __invert__(self: FN) -> FN:
        """
        Return a new FunctionNode that when evaulated, will invert the result of ``self``

        NOTE:
            This is generally expected to be a Boolean inversion, such as ~ (not) applied to a Numpy, Pandas, or Static-Frame objects.
        """
        return lambda *args, **kwargs: self(*args, **kwargs).__invert__()  # type: ignore

    @_wrap_unary
    def __abs__(self: FN) -> FN:
        """
        Return a new FunctionNode that when evaulated, will find the absolute value of the result of ``self``
        """
        return lambda *args, **kwargs: abs(self(*args, **kwargs))  # type: ignore

    # ---------------------------------------------------------------------------
    # all binary operators return a function; the _wrap_binary decorator then wraps this function in a FunctionNode definition and supplies appropriate doc args. Note both left and righ sides are wrapped in FNs to permit operations on constants

    @_wrap_binary
    def __add__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will add ``rhs`` to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) + self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __sub__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will subtract ``rhs`` to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) - self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __mul__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will multiply ``rhs`` by the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) * self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __truediv__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will divide ``rhs`` by the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) / self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __pow__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will divide raise the result of ``self`` by ``rhs``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) ** self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __radd__(self: FN, lhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will add ``rhs`` to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(lhs)(  # type: ignore
            *args, **kwargs
        ) + self.__class__(self)(*args, **kwargs)

    @_wrap_binary
    def __rsub__(self: FN, lhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will subtract ``rhs`` to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(lhs)(  # type: ignore
            *args, **kwargs
        ) - self.__class__(self)(*args, **kwargs)

    @_wrap_binary
    def __rmul__(self: FN, lhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will multiply ``rhs`` by the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(lhs)(  # type: ignore
            *args, **kwargs
        ) * self.__class__(self)(*args, **kwargs)

    @_wrap_binary
    def __rtruediv__(self: FN, lhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will divide ``rhs`` by the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(lhs)(  # type: ignore
            *args, **kwargs
        ) / self.__class__(self)(*args, **kwargs)

    # comparison operators, expected to return booleans
    @_wrap_binary
    def __eq__(self: FN, rhs: tp.Any) -> FN:  # type: ignore
        """
        Return a new FunctionNode will test if ``rhs``' equals the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) == self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __lt__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is less than the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) < self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __le__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is less than or equal to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) <= self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __gt__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is greater than the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) > self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __ge__(self: FN, rhs: tp.Any) -> FN:
        """
        Return a new FunctionNode will test if ``rhs``' is greater than or equal to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
            *args, **kwargs
        ) >= self.__class__(rhs)(*args, **kwargs)

    @_wrap_binary
    def __ne__(self: FN, rhs: tp.Any) -> FN:  # type: ignore
        """
        Return a new FunctionNode will test if ``rhs``' is not equal to the result of ``self``
        """
        return lambda *args, **kwargs: self.__class__(self)(  # type: ignore
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
        """Only implemented for PipeNode."""
        raise NotImplementedError()

    def __ror__(self: FN, lhs: FN) -> FN:
        """Only implemented for PipeNode."""
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
    """
    This encapsulates the node that will be used in a pipeline.

    It is not expected to be created directly, rather, through usage of ``pipe_node`` (and related) decorators.

    PipeNodes will be in (or move between) one of three states, depending on where it was created, or what the current state of pipeline evaluation is
    """

    class State(enum.Enum):
        """
        The current state of the PipeNode
        """

        FACTORY = "FACTORY"
        EXPRESSION = "EXPRESSION"
        PROCESS = "PROCESS"

    __slots__ = FunctionNode.__slots__ + (
        "_call_state",
        "_predecessor",
    )

    # ---------------------------------------------------------------------------
    def __init__(
        self: PN,
        function: tp.Any,
        *,
        doc_function: tp.Optional[tp.Callable] = None,
        doc_args: tp.Tuple[tp.Any, ...] = (),
        doc_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        call_state: tp.Optional[State] = None,
        predecessor: tp.Optional[PN] = None,
    ):
        super().__init__(
            function=function,
            doc_function=doc_function,
            doc_args=doc_args,
            doc_kwargs=doc_kwargs,
        )
        self._call_state = call_state
        self._predecessor = predecessor

    def __str__(self: PN) -> str:
        if self._call_state is PipeNode.State.FACTORY:
            return f"<PNF: {pretty_repr(self)}>"
        return f"<PN: {pretty_repr(self)}>"

    def __repr__(self: PN) -> str:
        if self._call_state is PipeNode.State.FACTORY:
            return f"<PNF: {pretty_repr(self)}>"
        return f"<PN: {pretty_repr(self)}>"

    def partial(self: PN, *args: str, **kwargs: str) -> PN:
        """
        Partialing PipeNodes is prohibited. Use ``pipe_node_factory`` (and related) decorators to pass in expression-level arguments.
        """
        raise NotImplementedError()

    # ---------------------------------------------------------------------------
    # pipe node properties

    @property
    def call_state(self: PN) -> tp.Optional["State"]:
        """The current call state of the Node"""
        return self._call_state

    @property
    def predecessor(self: PN) -> tp.Optional[PN]:
        """
        The PipeNode preceeding this Node in a pipeline. Can be None
        """
        return self._predecessor

    # ---------------------------------------------------------------------------
    # composition operators

    def __rshift__(self: PN, rhs: tp.Callable) -> PN:
        """Only implemented for FunctionNode."""
        raise NotImplementedError()

    def __rrshift__(self: PN, lhs: tp.Callable) -> PN:
        """Only implemented for FunctionNode."""
        raise NotImplementedError()

    def __lshift__(self: PN, rhs: tp.Callable) -> PN:
        """Only implemented for FunctionNode."""
        raise NotImplementedError()

    def __rlshift__(self: PN, lhs: tp.Callable) -> PN:
        """Only implemented for FunctionNode."""
        raise NotImplementedError()

    def __or__(self: PN, rhs: PN) -> PN:
        """
        Invokes ``rhs``, passing in ``self`` as the kwarg ``PREDECESSOR_PN``.
        """
        return rhs(**{PREDECESSOR_PN: self})

    def __ror__(self: PN, lhs: PN) -> PN:
        """
        Invokes ``lhs``, passing in ``self`` as the kwarg ``PREDECESSOR_PN``.
        """
        return self(**{PREDECESSOR_PN: lhs})

    # ---------------------------------------------------------------------------

    def __getitem__(self: PN, pn_input: tp.Any) -> tp.Any:
        """
        Invokes ``self``, passing in ``pn_input`` as the kwarg ``PN_INPUT``.

        NOTE:
            - If ``None``, will evaluate self with a default ``PipeNodeInput`` instance
            - If user desires for the initial input to be literally ``None``, use ``(**{PN_INPUT: None})`` instead.
        """
        pn_input = pn_input if pn_input is not None else PipeNodeInput()
        return self(**{PN_INPUT: pn_input})

    def __call__(self: PN, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """
        Call the wrapped function with args and kwargs.
        """
        if self._call_state is PipeNode.State.FACTORY:
            try:
                return self._function(*args, **kwargs)
            except Exception as e:
                raise _exception_with_cleaned_tb(e) from None

        if args or set(kwargs) - PIPE_NODE_KWARGS != set():
            raise ValueError(
                "Cannot call a PipeNode with args or non-pipeline kwargs! Perhaps you meant to use @pipe_node_factory?"
            )

        try:
            return self._function(**kwargs)
        except Exception as e:
            raise _exception_with_cleaned_tb(e) from None


# -------------------------------------------------------------------------------
# decorator utilities


def _broadcast(
    *,
    factory_args: tp.Tuple[tp.Any, ...],
    factory_kwargs: tp.Dict[str, tp.Any],
    processing_args: tp.Tuple[tp.Any, ...] = (),
    processing_kwargs: tp.Dict[str, tp.Any],
) -> tp.Tuple[tp.Tuple[tp.Any, ...], tp.Dict[str, tp.Any]]:
    """
    Factory args/kwargs are those given to pipe_node_factory at the expression level.
    Processing args/kwargs are those given as the initial input, and used to call all processing functions.

    After calling factory args with processing args, the result is used as core_callable args
    """
    core_callable_args = tuple(
        arg(*processing_args, **processing_kwargs) if isinstance(arg, PipeNode) else arg
        for arg in factory_args
    )

    core_callable_kwargs = {
        kw: arg(*processing_args, **processing_kwargs)
        if isinstance(arg, PipeNode)
        else arg
        for kw, arg in factory_kwargs.items()
    }

    return core_callable_args, core_callable_kwargs


def _core_logger(core_callable: tp.Callable) -> tp.Callable:
    """
    A decorator to provide output on the execution of each core callable call. Alternative decorators can be used to partial pipe_node_factory and pipe_node.
    """

    @functools.wraps(core_callable)
    def wrapped(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        print("|", str(core_callable), file=sys.stderr)
        return core_callable(*args, **kwargs)

    return wrapped


def _has_key_positions(*key_positions: KeyPostion) -> bool:
    """
    Returns whether or not key_positions is a list of key positions, or if it is just a single callable
    """
    return not bool(len(key_positions) == 1 and callable(key_positions[0]))


def is_unbound_self_method(
    core_callable: tp.Union[classmethod, staticmethod, tp.Callable],
    *,
    self_keyword: str,
) -> bool:
    """
    Inspects a given callable to determine if it's both unbound, and the first argument in its signature is ``self_keyword``
    """
    if isinstance(core_callable, types.MethodType):
        return False

    if isinstance(core_callable, (staticmethod, classmethod)):
        return False

    if isinstance(core_callable, functools.partial):
        return False

    argspec = inspect.getfullargspec(core_callable)
    return bool(argspec.args and argspec.args[0] == self_keyword)


def _pipe_kwarg_bind(
    *key_positions: KeyPostion,
) -> tp.Callable[[tp.Callable], tp.Callable]:
    """
    Binds a specific PN labels wrapped up in **kwargs to the first n positional arguments of the core callable
    """

    def decorator(core_callable: tp.Callable) -> tp.Callable:
        @functools.wraps(core_callable)
        def wrapped(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            target_args = [kwargs.pop(key) for key in key_positions]  # type: ignore
            target_kwargs = {
                k: v for k, v in kwargs.items() if k not in PIPE_NODE_KWARGS
            }
            return core_callable(*target_args, *args, **target_kwargs)

        return wrapped

    return decorator


class PipeNodeDescriptor:  # pylint: disable=too-few-public-methods
    """
    Wraps up ``pipe_node``/``pipe_node_factory`` behavior in a descriptor, where it will bind instance and owner to the core_callable, and then pass it along the pipeline
    """

    __slots__ = (
        "core_callable",
        "core_handler",
        "key_positions",
    )

    def __init__(
        self: PipeNodeDescriptorT,
        core_callable: tp.Callable,
        core_handler: HandlerT,
        key_positions: tp.Optional[tp.Tuple[KeyPostion, ...]] = None,
    ) -> None:
        self.core_callable = core_callable
        self.core_handler = core_handler
        self.key_positions = key_positions

    def __get__(
        self: PipeNodeDescriptorT,
        instance: tp.Any,
        owner: tp.Any,
    ) -> tp.Callable:
        """
        Returns a callable that will be bound to the instance/owner, and then passed along the pipeline.
        """
        core_callable: tp.Callable = self.core_callable.__get__(instance, owner)  # type: ignore
        if self.key_positions is not None:
            core_callable = _pipe_kwarg_bind(*self.key_positions)(core_callable)
        return self.core_handler(core_callable)


def _handle_descriptors_and_key_positions(
    *key_positions: KeyPostion,
    core_handler: HandlerT,
    self_keyword: str,
) -> tp.Union[
    PipeNodeDescriptor,
    HandlerT,
    tp.Callable[[tp.Callable], tp.Union[PipeNodeDescriptor, HandlerT]],
]:
    """
    We can return either a callable or a ``PipeNodeDescriptor``, OR, a decorator that when called,
    will return either a callable or a ``PipeNodeDescriptor``.
    """
    has_key_positions = _has_key_positions(*key_positions)

    # See if decorator was given no arguments, and received the core_callable directly.
    if not has_key_positions:
        final_callable = key_positions[0]
        assert callable(final_callable), (type(final_callable), final_callable)

        if is_unbound_self_method(final_callable, self_keyword=self_keyword):
            return PipeNodeDescriptor(final_callable, core_handler)

        return core_handler(final_callable)

    def decorator_wrapper(
        core_callable: tp.Callable,
    ) -> tp.Union[PipeNodeDescriptor, HandlerT]:
        if is_unbound_self_method(core_callable, self_keyword=self_keyword):
            return PipeNodeDescriptor(core_callable, core_handler, key_positions)

        final_callable = _pipe_kwarg_bind(*key_positions)(core_callable)
        return core_handler(final_callable)

    return decorator_wrapper


def _descriptor_factory(
    *key_positions: KeyPostion,
    decorator: tp.Callable,
    core_decorator: HandlerT,
    emulator: tp.Any,
) -> tp.Any:

    has_key_positions = _has_key_positions(*key_positions)

    class Descriptor:  # pylint: disable=too-few-public-methods
        def __init__(self, func: tp.Callable) -> None:
            self._func = func

        def __get__(self, instance: tp.Any, owner: tp.Any) -> tp.Callable:

            assert emulator is staticmethod or emulator is classmethod, emulator

            # Prefer this to partialing for prettier func reprs
            @functools.wraps(self._func)
            def func(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                if emulator is staticmethod:
                    return self._func(*args, **kwargs)
                return self._func(owner, *args, **kwargs)

            if has_key_positions:
                func = _pipe_kwarg_bind(*key_positions)(func)

            return decorator(func, core_decorator=core_decorator)

    if not has_key_positions:
        return Descriptor(key_positions[0])  # type: ignore

    return Descriptor


# -------------------------------------------------------------------------------
# decorators


def pipe_node_factory(
    *key_positions: KeyPostion,
    core_decorator: HandlerT = _core_logger,
    self_keyword: str = "self",
) -> tp.Union[tp.Callable, tp.Callable[[tp.Any], PipeNode]]:
    """
    Decorates a function to become a pipe node factory, that when given *expression-level* arguments, will return a ``PipeNode``

    This can either be used as a decorator, or a decorator factory, similar to ``functools.lru_cache``.

    **Examples**:

    >>> @pipe_node_factory
    >>> def func(a, b, **kwargs):
    >>>     pass
    >>> ...
    >>> func(1, 2) # This is now a PipeNode!

    >>> @pipe_node_factory()
    >>> def func(*, a, b):
    >>>     pass
    >>> ...
    >>> func(a=1, b=2) # This is now a PipeNode!

    >>> @pipe_node_factory(PN_INPUT, PREDECESSOR_RETURN)
    >>> def func(pn_input, previous_value, a, *, b):
    >>>     # pn_input will be given the PN_INPUT from the pipeline
    >>>     # prev will be given the PREDECESSOR_RETURN from the pipeline
    >>>     pass
    >>> ...
    >>> func(1, b=2) # This is now a PipeNode!

    >>> class Example:
    >>>     @pipe_node_factory(PN_INPUT, PREDECESSOR_RETURN)
    >>>     def method(self, pn_input, previous_value, a, *, b):
    >>>         pass
    >>> ...
    >>> Example().method(1, b=2) # This is now a PipeNode!

    >>> from functools import partial
    >>> español_pipe_node_factory = partial(pipe_node_factory, self_keyword="uno_mismo")
    >>> ...
    >>> class Ejemplo:
    >>>     @español_pipe_node_factory(PN_INPUT, PREDECESSOR_RETURN)
    >>>     def método(uno_mismo, pn_input, valor_anterior, a, *, b):
    >>>         pass
    >>> ...
    >>> Ejemplo().método(1, b=2) # Esto ahora es un PipeNode!

    Args:
        - ``key_positions``: either a single callable, or a list of keywords that will be positionally bound to the decorated function.
        - ``core_decorator``: a decorator that will be applied to the core_callable. This is typically a logger. By default, it will print to stdout.
        - ``self_keyword``: which keyword to look for when decorating instance methods.
    """

    def build_factory(core_callable: tp.Callable) -> PipeNode:
        decorated_core_callable = core_decorator(core_callable)

        def factory_f(*f_args: tp.Any, **f_kwargs: tp.Any) -> PipeNode:
            """This is the function returned by the decorator, used to create the PipeNode that resides in expressions after being called with arguments.

            f_args and f_kwargs are passed to the core_callable; if f_args or f_kwargs are PipeNode instances, they will be called with the processing args and kwargs (including PN_INPUT), either from process_f or (if innermost) from expression args.
            """
            if set(f_kwargs).intersection(PIPE_NODE_KWARGS) != set():
                raise ValueError(
                    f"Either you put a factory in a pipeline (i.e. not a pipe node), or your factory was given a reserved pipeline kwarg {tuple(PIPE_NODE_KWARGS)}."
                )

            def expression_f(**e_kwargs: tp.Any) -> PipeNode:
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

                def process_f(*p_args: tp.Any, **p_kwargs: tp.Any) -> tp.Any:
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
            factory_f,
            doc_function=core_callable,
            call_state=PipeNode.State.FACTORY,
        )

    return _handle_descriptors_and_key_positions(  # type: ignore
        *key_positions,
        core_handler=build_factory,
        self_keyword=self_keyword,
    )


def pipe_node(
    *key_positions: KeyPostion,
    core_decorator: HandlerT = _core_logger,
    self_keyword: str = "self",
) -> tp.Union[tp.Callable, PipeNode]:
    """
    Decorates a function to become a ``PipeNode`` that takes no expression-level args.

    This can either be used as a decorator, or a decorator factory, similar to ``functools.lru_cache``.

    **Examples**:

    >>> @pipe_node
    >>> def func(**kwargs):
    >>>     pass

    >>> @pipe_node()
    >>> def func():
    >>>     pass

    >>> @pipe_node(PN_INPUT)
    >>> def func(pn_input):
    >>>     pass

    >>> class Example:
    >>>     @pipe_node(PN_INPUT)
    >>>     def method(self, pn_input):
    >>>         pass

    >>> from functools import partial
    >>> español_pipe_node = partial(pipe_node, self_keyword="uno_mismo")
    >>> ...
    >>> class Ejemplo:
    >>>     @español_pipe_node(PN_INPUT)
    >>>     def método(uno_mismo, pn_input):
    >>>         pass

    Args:
        - ``key_positions``: either a single callable, or a list of keywords that will be positionally bound to the decorated function.
        - ``core_decorator``: a decorator that will be applied to the core_callable. This is typically a logger. By default, it will print to stdout.
        - ``self_keyword``: which keyword to look for when decorating instance methods.
    """

    def create_factory_and_call_once(core_callable: tp.Callable) -> PipeNode:
        # Create a factory and call it once with no args to get an expresion-level function
        pnf = pipe_node_factory(core_callable, core_decorator=core_decorator)

        if not callable(pnf):
            raise ValueError(f"{core_callable.__qualname__} requires an instance")

        return pipe_node_factory(core_callable, core_decorator=core_decorator)()  # type: ignore

    return _handle_descriptors_and_key_positions(  # type: ignore
        *key_positions,
        core_handler=create_factory_and_call_once,
        self_keyword=self_keyword,
    )


def classmethod_pipe_node_factory(
    *key_positions: KeyPostion, core_decorator: HandlerT = _core_logger
) -> tp.Callable:
    """
    Decorates a function to become a classmethod pipe node factory, that when given *expression-level* arguments, will return a ``PipeNode``

    This can either be used as a decorator, or a decorator factory, similar to ``functools.lru_cache``.

    This is a convenience method, that is the mental equivalent to this pseudo-code:

    >>> @classmethod
    >>> @pipe_node_factory(...)
    >>> def func(...)

    **Examples**:

    >>> @classmethod_pipe_node_factory
    >>> def func(cls, a, b, **kwargs):
    >>>     pass
    >>> ...
    >>> SomeClass.func(1, 2) # This is now a PipeNode!

    >>> @classmethod_pipe_node_factory()
    >>> def func(cls, *, a, b):
    >>>     pass
    >>> ...
    >>> SomeClass.func(a=1, b=2) # This is now a PipeNode!

    >>> @classmethod_pipe_node_factory(PN_INPUT, PREDECESSOR_RETURN)
    >>> def func(cls, pn_input, previous_value, a, *, b):
    >>>     # ``pn_input`` will be given the PN_INPUT from the pipeline
    >>>     # ``previous_value`` will be given the PREDECESSOR_RETURN from the pipeline
    >>>     pass
    >>> ...
    >>> SomeClass.func(1, b=2) # This is now a PipeNode!

    Args:
        - ``key_positions``: either a single callable, or a list of keywords that will be positionally bound to the decorated function.
        - ``core_decorator``: a decorator that will be applied to the core_callable. This is typically a logger. By default, it will print to stdout.
    """
    return _descriptor_factory(
        *key_positions,
        decorator=pipe_node_factory,
        core_decorator=core_decorator,
        emulator=classmethod,
    )


def classmethod_pipe_node(
    *key_positions: KeyPostion,
    core_decorator: HandlerT = _core_logger,
) -> tp.Union[tp.Callable, PipeNode]:
    """
    Decorates a function to become a classmethod ``PipeNode`` that takes no expression-level args.

    This can either be used as a decorator, or a decorator factory, similar to ``functools.lru_cache``.

    This is a convenience method, that is the mental equivalent to this pseudo-code:

    >>> @classmethod
    >>> @pipe_node(...)
    >>> def func(...)

    **Examples**:

    >>> @classmethod_pipe_node
    >>> def func(cls, **kwargs):
    >>>     pass

    >>> @classmethod_pipe_node()
    >>> def func(cls):
    >>>     pass

    >>> @classmethod_pipe_node(PN_INPUT)
    >>> def func(cls, pn_input):
    >>>     pass

    Args:
        - ``key_positions``: either a single callable, or a list of keywords that will be positionally bound to the decorated function.
        - ``core_decorator``: a decorator that will be applied to the core_callable. This is typically a logger. By default, it will print to stdout.
    """
    return _descriptor_factory(
        *key_positions,
        decorator=pipe_node,
        core_decorator=core_decorator,
        emulator=classmethod,
    )


def staticmethod_pipe_node_factory(
    *key_positions: KeyPostion, core_decorator: HandlerT = _core_logger
) -> tp.Callable:
    """
    Decorates a function to become a staticmethod pipe node factory, that when given *expression-level* arguments, will return a ``PipeNode``

    This can either be used as a decorator, or a decorator factory, similar to ``functools.lru_cache``.

    This is a convenience method, that is the mental equivalent to this pseudo-code:

    >>> @staticmethod
    >>> @pipe_node_factory(...)
    >>> def func(...)

    **Examples**:

    >>> @staticmethod_pipe_node_factory
    >>> def func(a, b, **kwargs):
    >>>     pass
    >>> ...
    >>> SomeClass.func(1, 2) # This is now a PipeNode!

    >>> @staticmethod_pipe_node_factory()
    >>> def func(*, a, b):
    >>>     pass
    >>> ...
    >>> SomeClass.func(a=1, b=2) # This is now a PipeNode!

    >>> @staticmethod_pipe_node_factory(PN_INPUT, PREDECESSOR_RETURN)
    >>> def func(pn_input, previous_value, a, *, b):
    >>>     # ``pn_input`` will be given the PN_INPUT from the pipeline
    >>>     # ``previous_value`` will be given the PREDECESSOR_RETURN from the pipeline
    >>>     pass
    >>> ...
    >>> SomeClass.func(1, b=2) # This is now a PipeNode!

    Args:
        - ``key_positions``: either a single callable, or a list of keywords that will be positionally bound to the decorated function.
        - ``core_decorator``: a decorator that will be applied to the core_callable. This is typically a logger. By default, it will print to stdout.
    """
    return _descriptor_factory(
        *key_positions,
        decorator=pipe_node_factory,
        core_decorator=core_decorator,
        emulator=staticmethod,
    )


def staticmethod_pipe_node(
    *key_positions: KeyPostion,
    core_decorator: HandlerT = _core_logger,
) -> tp.Union[tp.Callable, PipeNode]:
    """
    Decorates a function to become a staticmethod ``PipeNode`` that takes no expression-level args.

    This can either be used as a decorator, or a decorator factory, similar to ``functools.lru_cache``.

    This is a convenience method, that is the mental equivalent to this pseudo-code:

    >>> @staticmethod
    >>> @pipe_node(...)
    >>> def func(...)

    **Examples**:

    >>> @staticmethod_pipe_node
    >>> def func(**kwargs):
    >>>     pass

    >>> @staticmethod_pipe_node()
    >>> def func():
    >>>     pass

    >>> @staticmethod_pipe_node(PN_INPUT)
    >>> def func(pn_input):
    >>>     pass

    Args:
        - ``key_positions``: either a single callable, or a list of keywords that will be positionally bound to the decorated function.
        - ``core_decorator``: a decorator that will be applied to the core_callable. This is typically a logger. By default, it will print to stdout.
    """
    return _descriptor_factory(
        *key_positions,
        decorator=pipe_node,
        core_decorator=core_decorator,
        emulator=staticmethod,
    )


# -------------------------------------------------------------------------------
# PipeNodeInput


class PipeNodeInput:
    """
    PipeNode input to support store and recall; subclassable to expose other attributes and parameters.
    """

    def __init__(self: PNI) -> None:
        self._store: tp.Dict[str, tp.Any] = {}

    def store(self: PNI, key: str, value: tp.Any) -> None:
        """Store ``key`` and ``value`` in the underlying store."""
        if key in self._store:
            raise KeyError("cannot store the same key", key)
        self._store[key] = value

    def recall(self: PNI, key: str) -> tp.Any:
        """Recall ``key`` from the underlying store. Can raise an ``KeyError``"""
        return self._store[key]

    @property
    def store_items(self: PNI) -> tp.ItemsView[str, tp.Any]:
        """Return an items view of the underlying store."""
        return self._store.items()


# -------------------------------------------------------------------------------
# Utility PipeNodes


@pipe_node_factory(PN_INPUT, PREDECESSOR_RETURN)
def store(pn_input: PipeNodeInput, previous_value: tp.Any, label: str) -> tp.Any:
    """
    Store ``previous_value`` (the value returned from the previous ``PipeNode``) to ``pn_input`` under ``label``. Forward ``previous_value``.
    """
    pn_input.store(label, previous_value)
    return previous_value


@pipe_node_factory(PN_INPUT)
def recall(pn_input: PipeNodeInput, label: str) -> tp.Any:
    """
    Recall ``label`` from ``pn_input```` and return it. Can raise an ``KeyError``.
    """
    return pn_input.recall(label)


@pipe_node_factory()
def call(*pns: PipeNode) -> tp.Any:
    """
    Since ``pns`` are all ``PipeNodes``, they will all be evaluated before passed in as values to this function.

    This is called broadcasting.

    After they have all been evaluated, return the last result.
    """
    return pns[-1]  # the last result is returned
