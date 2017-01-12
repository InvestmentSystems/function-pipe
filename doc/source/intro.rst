

Introduction
==================================

The function-pipe module defines the class FunctionNode (FN) and decorators to create the derived-class PipeNode (PN). FNs are wrappers of callables that permit returning new FNs after applying operators, composing callables, or partialing. This supports the flexible combination of functions in a lazy and declarative manner.

PipeNodes (PNs) are FNs prepared for extended function composition. PNs, through a decorator-provided, two-stage calling mechanism, expose to wrapped functions both predecessor output and a common initial input. Rather than strictly linear pipelines, sequences of PNs can be stored and reused; PNs can be provided as arguments to other PNs; and results from PNs can be stored for later recall in the same pipeline.


Getting Started
----------------

FunctionNode and PipeNode are abstract tools for linking functions, applicable in many domains. The best way to get to know them is to follow some examples and experiment. This documentation provide examples in a number of domains, including processing strings and Pandas DataFrames.


Installation
------------------

A standard setuptools installer will be available shortly.

For now, source code can be obtained here:
https://github.com/InvestmentSystems/function-pipe


History
--------

The function-pipe tools were developed within Investment Systems, the core development team of Research Affiliates, LLC. Many of the approaches implemented were first created by Max Moroz in 2012. Christopher Ariza subsequently refined and extended those approaches into the current models of FunctionNode and PipeNode. The first public release of function-pipe was in January 2017.



