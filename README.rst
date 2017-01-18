function-pipe
=============

The function-pipe Python module defines the class FunctionNode (FN) and decorators to create the derived-class PipeNode (PN). FNs are wrappers of callables that permit returning new FNs after applying operators, composing callables, or partialing. This supports the flexible combination of functions in a lazy and declarative manner.

PipeNodes (PNs) are FNs prepared for extended function composition or dataflow programming. PNs, through a decorator-provided, two-stage calling mechanism, expose to wrapped functions both predecessor output and a common initial input. Rather than strictly linear pipelines, sequences of PNs can be stored and reused; PNs can be provided as arguments to other PNs; and results from PNs can be stored for later recall in the same pipeline.

Code: https://github.com/InvestmentSystems/function-pipe

Docs: http://function-pipe.readthedocs.io

Packages: https://pypi.python.org/pypi/function-pipe

