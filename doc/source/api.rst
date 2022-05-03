API
***

function_pipe
=============

.. currentmodule:: function_pipe

.. autoclass:: FunctionNode
    :members:
    :special-members: __init__, __call__, __neg__, __invert__, __abs__, __add__, __sub__, __mul__, __truediv__, __pow__, __radd__, __rsub__, __rmul__, __rtruediv__, __eq__, __lt__, __le__, __gt__, __ge__, __ne__, __rshift__, __rrshift__, __lshift__, __rlshift__, __or__, __ror__
    :member-order: bysource


.. autoclass:: PipeNode
    :members:
    :exclude-members: State
    :special-members: __or__, __ror__, __getitem__, __call__
    :member-order: bysource


.. autoclass:: PipeNodeInput
   :members:
   :member-order: bysource

.. autofunction:: pipe_node
.. autofunction:: pipe_node_factory
.. autofunction:: compose
.. autofunction:: classmethod_pipe_node
.. autofunction:: classmethod_pipe_node_factory
.. autofunction:: staticmethod_pipe_node
.. autofunction:: staticmethod_pipe_node_factory
.. function:: store(pni: PipeNodeInput, ret_val: tp.Any, label: str) -> tp.Any:

    Store ``ret_val`` (the value returned from the previous ``PipeNode``) to ``pni`` under ``label``. Forward ``ret_val``.

.. function:: recall(pni: PipeNodeInput, label: str) -> tp.Any:

    Recall ``label`` from ``pni```` and return it. Can raise an ``KeyError``

.. function:: call(*pns: PipeNode) -> tp.Any

    Broadcasts ``pns``, and returns the result of ``pns[-1]``

    Since ``pns`` are all ``PipeNodes``, they will all be evaluated before passed in as values to this function.
