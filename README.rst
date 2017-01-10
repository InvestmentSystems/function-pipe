function-pipe
=============

The function-pipe module defines the class FunctionNode (FN) and decorators to create the derived class PipeNode (PN). FNs are wrappers of callables that permit returning new FNs from operator application and composition with other FNs. This supports combining functions in a lazy and declarative manner. PipeNodes (PNs) are FNs prepared for extended function composition. PNs expose to wrapped functions both predecessor output and a common initial input shared by all nodes. Rather than strictly linear pipelines, sequences of PNs can be stored and reused; PNs can be provided as arguments to other PNs; and results from PNs can be stored for later recall later in the same pipeline.


Code: https://github.com/InvestmentSystems/function-pipe
Docs: http://function-pipe.readthedocs.io



