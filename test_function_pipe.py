import pytest
import function_pipe as fpn

def test_prematurely_called_pn():
    @fpn.pipe_node
    def init(**kwargs):
        return 42

    @fpn.pipe_node
    def func(**kwargs):
        pass

    pni = fpn.PipeNodeInput()

    with pytest.raises(RuntimeError):
        expr = (init | func())

    # Correct
    expr = (init | func)
    expr[pni]


def test_non_initialized_pn_factory():

    @fpn.pipe_node
    def init(**kwargs):
        return 42


    @fpn.pipe_node_factory
    def print_kwargs_factory(**kwargs):
        pass

    pni = fpn.PipeNodeInput()

    with pytest.raises(RuntimeError):
        (init | print_kwargs_factory)[pni]

    # Correct
    (init | print_kwargs_factory())[pni]
