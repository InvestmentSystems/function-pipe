Intuitively Understanding Pipelines
***********************************

.. admonition:: Tutorial Alias

   PN: pipe node

This tutorial will teach the foundational concepts of ``function_pipe`` PN pipelines through clear and intuitive steps. After reading, you will:

      * Know how to build and use a PN and/or PN pipeline
      * Understand the different between the **creation** and **evaluation** phase of PN
      * Understand how to link PNs together
      * Understand what a PN input is, and how to share data across PNs
      * Be able to debug issues in your own PNs

Intro
=====

Function pipelines happen in two stages: **creation** & **evaluation**.

**Creation** is the step in which a pipeline is defined, understood as either a single PN, or multiple PNs chained together using the ``|`` operator. Here is a pseudo-code example of this:

.. code:: python

   pipeline = (pn_a | pn_b | pn_c | ...)

   # OR

   pipeline = pn

**Evaluation** is the step in which the pipeline is actually called, where the function code inside each PN is actually run:

.. code:: python

   pipeline["initial input"] # Evaluate the pipeline by using __getitem__, and passing in some initial input

Visualizing the Distinction Between Creation & Evaluation
=========================================================

To get started, we will create two simple PNs, put them into a pipeline expression, and then evaluate that expression. **Creation** followed by **evaluation**.

To do this, we will use the ``fpn.pipe_node`` decorator, and define methods which take ``**kwargs``. (``**kwargs`` will be explained later!)

.. code:: python
   :class: copy-button

   import function_pipe as fpn  # Import convention!

   @fpn.pipe_node
   def pipe_node_1(**kwargs):
      print("pipe_node_1 has been evaluated")

   @fpn.pipe_node
   def pipe_node_2(**kwargs):
      print("pipe_node_2 has been evaluated")

   print("Start creation")
   pipeline = (pipe_node_1 | pipe_node_2)
   print("End creation")

   print("Start pipeline evaluation")
   pipeline[None]
   print("End pipeline evaluation")


Now, let's see the output that happens were we to run the previous code.

.. code::

   Start creation
   End creation
   Start pipeline evaluation
   | <function pipe_node_1 at 0x7f582c428ca0>
   pipe_node_1 has been evaluated
   | <function pipe_node_2 at 0x7f582c428b80>
   pipe_node_2 has been evaluated
   End pipeline evaluation

As you can see, none of the PNs are called (**evaluated**) until the pipeline expression itself was **created** and then invoked.


What Is The Deal With Kwargs
============================

In the previous example, we used ``**kwargs`` on each function (if we hadn't, the code would have failed!) Why did we need this, and what are they? Let's investigate!

To investigate, we will build up a slightly longer pipeline, and expand the nodes to return some values

.. code:: python
   :class: copy-button

   @fpn.pipe_node
   def pipe_node_1(**kwargs):
      print(kwargs)
      return 1

   @fpn.pipe_node
   def pipe_node_2(**kwargs):
      print(kwargs)
      return 2

   @fpn.pipe_node
   def pipe_node_3(**kwargs):
      print(kwargs)
      return 3

   pipeline = (pipe_node_1 | pipe_node_2 | pipe_node_3)
   assert pipeline["original_input"] == 3

   print(f"repr(pipeline) = '{repr(pipeline)}'")

Running the above code will produce the following output:

.. code:: python

   | <function pipe_node_1 at 0x7f582cceb700>
   {"pn_input": "original_input"}
   | <function pipe_node_2 at 0x7f582c2d30d0>
   {"pn_input": "original_input", "predecessor_pn": <PN: pipe_node_1>, "predecessor_return": 1}
   | <function pipe_node_3 at 0x7f582c33b820>
   {"pn_input": "original_input", "predecessor_pn": <PN: pipe_node_1 | pipe_node_2>, "predecessor_return": 2}
   repr(pipeline) = '<PN: pipe_node_1 | pipe_node_2 | pipe_node_3>'

There are a few things happening here worth observing.

1) Every node is given the kwarg ``pn_input``.
2) Each node (except the first), is given the kwargs ``predecessor_pn`` and ``predecessor_return``

The first node is special. In the context of the pipeline it lives in, there are no PNs preceding it, hence ``predecessor_pn`` and ``predecessor_return`` are not passed in!

For every other node, it is initiutive what the values of ``predecessor_pn`` and ``predecessor_return`` will be. They contain the node instance of the one before, and the return value of that node once it's evaluated.

As we can observe on ``pipe_node_3``, the repr of ``predecessor_pn`` shows how it's predecessor is actually a pipeline of PNs instead of a single PN. Additionally, printing the repr of ``pipeline`` shows how it is a pipeline of multiple PNs.

.. note::
   From now on, we will refer to the three strings above by their symbolic constant handles in the **function_pipe** module. They are ``fpn.PN_INPUT``, ``fpn.PREDECESSOR_PN``, and ``fpn.PREDECESSOR_RETURN``, respectively.

Using the Kwargs
================

Now that we know what will be passed in through each PN's ``**kwargs`` based on where it is in the pipeline, let's write some code that takes advantage of that.

.. code:: python
   :class: copy-button

   @fpn.pipe_node
   def multiply_input_by_2(**kwargs):
      return kwargs[fpn.PN_INPUT] * 2

   @fpn.pipe_node
   def add_7(**kwargs):
      return kwargs[fpn.PREDECESSOR_RETURN] + 7

   @fpn.pipe_node
   def divide_by_3(**kwargs):
      return kwargs[fpn.PREDECESSOR_RETURN] / 3

   pipeline_1 = (multiply_input_by_2 | add_7 | divide_by_3)
   assert pipeline_1[12] == (((12 * 2) + 7) / 3)

   pipeline_2 = (multiply_input_by_2 | divide_by_3 | add_7)
   assert pipeline_2[12] == (((12 * 2) / 3) + 7)

As you can see, PNs have the ability to use the return values from their predecessors, or the ``fpn.PN_INPUT`` whenever they need to.

You can also observe that ``pipeline_2`` reversed the order of the latter two PNs from their order in ``pipeline_1``. This worked seamlessly, since each of the PNs was accessing information from the predecessor's return value. Had we tried something like:

.. code:: python
   :class: copy-button

   pipeline_3 = (add_7 | multiply_input_by_2 | divide_by_3)
   pipeline_3[12]

it would have failed, since the first PN is *never* given ``fpn.PREDECESSOR_RETURN`` as a kwarg.

.. note::
   ``fpn.PREDECESSOR_PN`` is a kwarg that is almost never used in regular PNs or pipelines. If you are reaching for this kwarg, you are probably doing something wrong! It's primary purpose is to ensure the internals of the **function_pipe.PipeNode** module are working properly, not for use by end users.

Hiding the Kwargs
=================

Now that we know how to use ``**kwargs``, we can see that manually extracting the pipeline kwargs we care about each time is not good! On top of that, it's highly undesirable to require the signature of all PNs to accept arbitrary ``**kwargs``.

Lucky for us, the ``fpn.pipe_node`` decorator can be optionally given the desired kwargs we want to positionally bind in the actual function signature.

.. code:: python
   :class: copy-button

   # Bind the first positional argument
   @fpn.pipe_node(fpn.PN_INPUT)
   def multiply_input_by_2(pn_input):
      return pn_input * 2

   # Bind the first positional argument
   @fpn.pipe_node(fpn.PREDECESSOR_RETURN)
   def add_7(previous_value):
      return previous_value + 7

   # Bind the first and second positional arguments
   @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
   def divide_by_3_add_pn_input(pn_input, previous_value):
      return (previous_value / 3) + pn_input

   @fpn.pipe_node() # Bind no arguments
   def nothing_is_bound():
      pass

   pipeline = (
      nothing_is_bound
      | multiply_input_by_2
      | add_7
      | divide_by_3_add_pn_input
   )
   assert pipeline[12] == ((((12 * 2) + 7) / 3) + 12)

Ah. That's much better. It clears up the function signature, and makes it clear what each PN function needs in order to process properly.

To restate what's happening, arguments given to the decorator will be extracted from the pipeline, and implicitly passed in as the first positional arguments defined in the function signature.

What About Other Arguments
==========================

So far, we have most of the basics. However, there is one essential use case missing: how do I define additional arguments on my function? Let's say instead of a PN called ``add_7``, I want to have a PN called ``add``, that takes an argument that will be added to the predecessor return value. Here's a pseudo-code example:

.. code:: python

   @fpn.pipe_node(fpn.PREDECESSOR_RETURN)
   def add(previous_value, value_to_add):
      return previous_value + value_to_add

   pipeline = (... | ... | add(13) | .. )

Ideally, there should be a mechanism that allows the user *bind* (or *partial*) custom args & kwargs to give their pipelines all the flexibility needed.

Welcome To the Factory
======================

Thankfully, such a mechanism exists: it's called ``fpn.pipe_node_factory``. This is the other key decorator we need to know for building PNs.

The previous example would work exactly as expected had we replaced the ``fpn.pipe_node`` decorator with the ``fpn.pipe_node_factory`` decorator!

.. code:: python
   :class: copy-button

   @fpn.pipe_node(fpn.PN_INPUT)
   def init(pn_input):
      return pn_input

   @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
   def add(previous_value, value_to_add):
      return previous_value + value_to_add

   pipeline = (init | add(3) | add(4.2) | add(-2003))
   assert pipeline[0] == (0 + 3 + 4.2 + -2003)

To reiterate what's happening here, the ``fpn.pipe_node_factory`` decorates the method in such way it can be thought of as a factory that builds PNs. This is essential, since every element in a pipeline **must** be a PN! The PN factories allow us to used *bound* (or *partialed*) PN with arbitrary args/kwargs.


A Common Factory Mistake
========================

A common failure when using ``fpn.pipe_node_factory`` is forgetting to call the decorator before it's put into the pipeline!

Building on the previous example, let's see what happens if we forgot to add an
argument to ``add``.

.. code:: python
   :class: copy-button

   @fpn.pipe_node(fpn.PN_INPUT)
   def init(pn_input):
      return pn_input

   @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
   def add(previous_value, value_to_add):
      return previous_value + value_to_add

   # Uh-oh! One of the `add` pn factories was not given its required argument!
   pipeline = (init | add(3) | add(4.2) | add)

Let's see the failure message this will raise:

.. code:: python

   ---------------------------------------------------------------------------
   ValueError                                Traceback (most recent call last)
   ...
   ValueError: Either you put a factory in a pipeline (i.e. not a pipe node), or your factory was given a reserved pipeline kwarg ('pn_input', 'predecessor_pn', 'predecessor_return').

This failure should make sense now! Every node in a pipline **must** be a PN. Since ``add`` was not given a factory argument, it was a *PN factory*, **not** a PN.


PN Input (pni)
==============

.. admonition:: Code Alias

   pni: pn_input (argument conventionally bound to ``fpn.PN_INPUT``)

Up until now, the usage of ``pni`` (i.e. the argument conventionally bound to ``fpn.PN_INPUT``) has been a relatively diverse. This is because ``fpn.PN_INPUT`` refers to the initial input to the pipeline, and as such, can be any value. For these simple examples, I have been providing integers, but real-world cases typically rely on the ``fpn.PipeNodeInput`` class.

``fpn.PipeNodeInput`` is a subclassable object, which has the ability to:

   1. Store results from previous PNs
   2. Recall values from previous PNs
   3. Share state across PNs.

Let's observe the following example, where we subclass ``fpn.PipeNodeInput`` in order to share some state accross PNs.

.. code:: python
   :class: copy-button

   class PNI(fpn.PipeNodeInput):
      def __init__(self, state):
         super().__init__()
         self.state = state

   pni_12 = PNI(12)

   @fpn.pipe_node(fpn.PN_INPUT)
   def pipe_node_1(pni):
      return pni.state * 2

   @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
   def pipe_node_2(pni, previous_value):
      return (pni.state * previous_value) / 33

   @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
   def pipe_node_3(pni, previous_value):
      return (previous_value ** pni.state) -16

   pipeline = (pipe_node_1 | pipe_node_2 | pipe_node_3)
   assert pipeline[pni_12] == ((((12 * (12 * 2)) / 33) ** 12) - 16)

This is also a good opportunity to highlight how pipeline expressions can be easily reused to provide different results when given different inital inputs. Using the above example, giving a different ``pni`` will give us a totally different result:

.. code:: python
   :class: copy-button

   pni_99 = PNI(99)
   assert pipeline[pni_99] == ((((99 * (99 * 2)) / 33) ** 99) - 16)
   assert pipeline[pni_99] != pipeline[pni_12]

Store & Recall
==============

One of the main benefits to using a ``fpn.PipeNodeInput`` subclass, is the ability to use ``fpn.store`` and ``fpn.recall``. These utility methods will store & recall results from a cache privately stored on the ``pni``.

.. code:: python
   :class: copy-button

   @fpn.pipe_node()
   def returns_12345():
      return 12345

   @fpn.pipe_node(fpn.PREDECESSOR_RETURN)
   def double_previous(previous_value):
      return previous_value * 2

   @fpn.pipe_node(fpn.PREDECESSOR_RETURN)
   def return_previous(previous_value):
      return previous_value

   pni = fpn.PipeNodeInput()

   pipeline_1 = (
      returns_12345
      | fpn.store("first_result")
      | double_previous
      | fpn.store("second_result")
   )
   pipeline_1[pni]

   pipeline_2 = (fpn.recall("first_result") | return_previous)
   assert pipeline_2[pni] == 12345

   pipeline_3 = (fpn.recall("second_result") | return_previous)
   assert pipeline_3[pni] == (12345 * 2)

As you can see, once results have been stored using ``fpn.store``, they are retrievable using ``fpn.recall`` for any other pipeline **that is evaluated with that same pni**!

Additionally, you can see that ``fpn.store`` and ``fpn.recall`` simply forward along the previous return values so that they can be seamlessly inserted anywhere into a pipeline.

.. note::
   ``fpn.store`` and ``fpn.recall`` only work when the initial input is a valid instance or subclass instance of ``fpn.PipeNodeInput``.


Advanced - Instance/Class/Static Methods
========================================

The final section in this tutorial explains the tools needed for turning ``classmethods`` and ``staticmethods`` into PNs. To do this, we can take advantage of special classmethod/staticmethod tools built into the **function_pipe** library!

.. note::
   Normal "instance" methods (i.e. functions that expect self (i.e. the instance) passed in as the first argument) work exactly as expected with the ``fpn.pipe_node`` and ``fpn.pipe_node_factory`` decorators, as long as the name of the argument is "self".

Building on everything we've seen so far, let's take a look at the class below, which demonstrates usage of ``fpn.classmethod_pipe_node``, ``fpn.classmethod_pipe_node_factory``, ``fpn.staticmethod_pipe_node`` and ``fpn.staticmethod_pipe_node_factory``.

.. code:: python
   :class: copy-button

   class Operations:
      STATE = 1

      def __init__(self, state):
         self.state = state

      @fpn.pipe_node
      def operation_1(self, **kwargs):
         # This works as expected, since the first argument is "self"
         return self.state + kwargs[fpn.PN_INPUT].state

      @fpn.classmethod_pipe_node
      def operation_2(cls, **kwargs):
         return cls.STATE + kwargs[fpn.PN_INPUT].state

      @fpn.staticmethod_pipe_node
      def operation_3(**kwargs):
         return kwargs[fpn.PN_INPUT].state

      @fpn.pipe_node_factory
      def operation_4(self, user_arg, *, user_kwarg, **kwargs):
         return (self.state + user_arg - user_kwarg) * kwargs[fpn.PN_INPUT].state

      @fpn.classmethod_pipe_node_factory
      def operation_5(cls, user_arg, *, user_kwarg, **kwargs):
         return (cls.STATE + user_arg - user_kwarg) * kwargs[fpn.PN_INPUT].state

      @fpn.staticmethod_pipe_node_factory
      def operation_6(user_arg, *, user_kwarg, **kwargs):
         return (user_arg - user_kwarg) * kwargs[fpn.PN_INPUT].state

      @fpn.pipe_node(fpn.PN_INPUT)
      def operation_7(self, pni):
         return (self.state + pni.state) * 2

      @fpn.classmethod_pipe_node_factory(fpn.PREDECESSOR_RETURN)
      def operation_8(cls, previous_value, user_arg, *, user_kwarg):
         return (cls.STATE + user_arg - user_kwarg) * previous_value

      @fpn.staticmethod_pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
      def operation_9(pni, previous_value):
         return (pni.state - previous_value) ** 2

   class PNI(fpn.PipeNodeInput):
      def __init__(self, state):
         super().__init__()
         self.state = state

   pni = PNI(-99)

   op = Operations(2)

   pipeline = (
         # The first three are PNs!
         op.operation_1
         | op.operation_2
         | op.operation_3
         # The second three are PN factories!
         | op.operation_4(10, user_kwarg=11)
         | op.operation_5(12, user_kwarg=13)
         | op.operation_6(14, user_kwarg=15)
         # The rest are PNs (except `operation_8`)
         | op.operation_7
         | op.operation_8(16, user_kwarg=17)
         | op.operation_9
   )

   assert pipeline[pni] == 9801 # Good luck figuring that one out ;)

To help explain the decorators a bit more, here is a quick pseudo-code example showing an alternative way to understand them:

.. code:: python

   @fpn.classmethod_pipe_node

   # Behaves like you think this would:

   @classmethod
   @fpn.pipe_node

   # ------------------------------------------------------------

   @fpn.staticmethod_pipe_node_factory

   # Behaves like you think this would:

   @staticmethod
   @fpn.pipe_node_factory

   # etc...

Conclusion
==========

After going through this tutorial, you should now have an understanding of:

   - The **creation** and **evaluation** stages of a pipeline
   - The ``fpn.pipe_node`` decorator, and when to use it
   - The ``fpn.pipe_node_factory`` decorator, and when to use it
   - How to positionally bind the first argument(s) of a pipeline to ``fpn.PN_INPUT`` and/or ``fpn.PREDECESSOR_RETURN``.
   - How to use ``fpn.store`` and ``fpn.recall`` to store and recall results from a pipeline.
   - How to use ``fpn.PipeNodeInput``.
   - How to make instance methods, classmethods, and staticmethods into PNs.

Here is all of the code examples we have seen so far:

.. code:: python
   :class: copy-button

   # Example 1:

   import function_pipe as fpn  # Import convention!

   @fpn.pipe_node
   def pipe_node_1(**kwargs):
      print("pipe_node_1 has been evaluated")

   @fpn.pipe_node
   def pipe_node_2(**kwargs):
      print("pipe_node_2 has been evaluated")

   print("Start creation")
   pipeline = (pipe_node_1 | pipe_node_2)
   print("End creation")

   print("Start pipeline evaluation")
   pipeline[None]
   print("End pipeline evaluation")

   # Example 2:

   @fpn.pipe_node
   def pipe_node_1(**kwargs):
      print(kwargs)
      return 1

   @fpn.pipe_node
   def pipe_node_2(**kwargs):
      print(kwargs)
      return 2

   @fpn.pipe_node
   def pipe_node_3(**kwargs):
      print(kwargs)
      return 3

   pipeline = (pipe_node_1 | pipe_node_2 | pipe_node_3)
   assert pipeline["original_input"] == 3

   print(f"repr(pipeline) = '{repr(pipeline)}'")

   # Example 3:

   @fpn.pipe_node
   def multiply_input_by_2(**kwargs):
      return kwargs[fpn.PN_INPUT] * 2

   @fpn.pipe_node
   def add_7(**kwargs):
      return kwargs[fpn.PREDECESSOR_RETURN] + 7

   @fpn.pipe_node
   def divide_by_3(**kwargs):
      return kwargs[fpn.PREDECESSOR_RETURN] / 3

   pipeline_1 = (multiply_input_by_2 | add_7 | divide_by_3)
   assert pipeline_1[12] == (((12 * 2) + 7) / 3)

   pipeline_2 = (multiply_input_by_2 | divide_by_3 | add_7)
   assert pipeline_2[12] == (((12 * 2) / 3) + 7)

   # Example 4:

   pipeline_3 = (add_7 | multiply_input_by_2 | divide_by_3)

   try:
      pipeline_3[12]
   except KeyError as e:
      print(e)

   # Example 5:

   # Bind the first positional argument
   @fpn.pipe_node(fpn.PN_INPUT)
   def multiply_input_by_2(pn_input):
      return pn_input * 2

   # Bind the first positional argument
   @fpn.pipe_node(fpn.PREDECESSOR_RETURN)
   def add_7(previous_value):
      return previous_value + 7

   # Bind the first and second positional arguments
   @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
   def divide_by_3_add_pn_input(pn_input, previous_value):
      return (previous_value / 3) + pn_input

   @fpn.pipe_node() # Bind no arguments
   def nothing_is_bound():
      pass

   pipeline = (
      nothing_is_bound
      | multiply_input_by_2
      | add_7
      | divide_by_3_add_pn_input
   )
   assert pipeline[12] == ((((12 * 2) + 7) / 3) + 12)

   # Example 6:

   @fpn.pipe_node(fpn.PN_INPUT)
   def init(pn_input):
      return pn_input

   @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
   def add(previous_value, value_to_add):
      return previous_value + value_to_add

   pipeline = (init | add(3) | add(4.2) | add(-2003))
   assert pipeline[0] == (0 + 3 + 4.2 + -2003)

   # Example 7:

   @fpn.pipe_node(fpn.PN_INPUT)
   def init(pn_input):
      return pn_input

   @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
   def add(previous_value, value_to_add):
      return previous_value + value_to_add

   # Uh-oh! One of the `add` pn factories was not given its required argument!
   try:
      pipeline = (init | add(3) | add(4.2) | add)
   except ValueError as e:
      print(e)

   # Example 8:

   class PNI(fpn.PipeNodeInput):
      def __init__(self, state):
         super().__init__()
         self.state = state

   pni_12 = PNI(12)

   @fpn.pipe_node(fpn.PN_INPUT)
   def pipe_node_1(pni):
      return pni.state * 2

   @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
   def pipe_node_2(pni, previous_value):
      return (pni.state * previous_value) / 33

   @fpn.pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
   def pipe_node_3(pni, previous_value):
      return (previous_value ** pni.state) -16

   pipeline = (pipe_node_1 | pipe_node_2 | pipe_node_3)
   assert pipeline[pni_12] == ((((12 * (12 * 2)) / 33) ** 12) - 16)

   # Example 9:

   pni_99 = PNI(99)
   assert pipeline[pni_99] == ((((99 * (99 * 2)) / 33) ** 99) - 16)
   assert pipeline[pni_99] != pipeline[pni_12]

   # Example 10:

   @fpn.pipe_node()
   def returns_12345():
      return 12345

   @fpn.pipe_node(fpn.PREDECESSOR_RETURN)
   def double_previous(previous_value):
      return previous_value * 2

   @fpn.pipe_node(fpn.PREDECESSOR_RETURN)
   def return_previous(previous_value):
      return previous_value

   pni = fpn.PipeNodeInput()

   pipeline_1 = (
      returns_12345
      | fpn.store("first_result")
      | double_previous
      | fpn.store("second_result")
   )
   pipeline_1[pni]

   pipeline_2 = (fpn.recall("first_result") | return_previous)
   assert pipeline_2[pni] == 12345

   pipeline_3 = (fpn.recall("second_result") | return_previous)
   assert pipeline_3[pni] == (12345 * 2)

   # Example 11:

   class Operations:
      STATE = 1

      def __init__(self, state):
         self.state = state

      @fpn.pipe_node
      def operation_1(self, **kwargs):
         # This works as expected, since the first argument is "self"
         return self.state + kwargs[fpn.PN_INPUT].state

      @fpn.classmethod_pipe_node
      def operation_2(cls, **kwargs):
         return cls.STATE + kwargs[fpn.PN_INPUT].state

      @fpn.staticmethod_pipe_node
      def operation_3(**kwargs):
         return kwargs[fpn.PN_INPUT].state

      @fpn.pipe_node_factory
      def operation_4(self, user_arg, *, user_kwarg, **kwargs):
         return (self.state + user_arg - user_kwarg) * kwargs[fpn.PN_INPUT].state

      @fpn.classmethod_pipe_node_factory
      def operation_5(cls, user_arg, *, user_kwarg, **kwargs):
         return (cls.STATE + user_arg - user_kwarg) * kwargs[fpn.PN_INPUT].state

      @fpn.staticmethod_pipe_node_factory
      def operation_6(user_arg, *, user_kwarg, **kwargs):
         return (user_arg - user_kwarg) * kwargs[fpn.PN_INPUT].state

      @fpn.pipe_node(fpn.PN_INPUT)
      def operation_7(self, pni):
         return (self.state + pni.state) * 2

      @fpn.classmethod_pipe_node_factory(fpn.PREDECESSOR_RETURN)
      def operation_8(cls, previous_value, user_arg, *, user_kwarg):
         return (cls.STATE + user_arg - user_kwarg) * previous_value

      @fpn.staticmethod_pipe_node(fpn.PN_INPUT, fpn.PREDECESSOR_RETURN)
      def operation_9(pni, previous_value):
         return (pni.state - previous_value) ** 2

   class PNI(fpn.PipeNodeInput):
      def __init__(self, state):
         super().__init__()
         self.state = state

   pni = PNI(-99)

   op = Operations(2)

   pipeline = (
         # The first three are PNs!
         op.operation_1
         | op.operation_2
         | op.operation_3
         # The second three are PN factories!
         | op.operation_4(10, user_kwarg=11)
         | op.operation_5(12, user_kwarg=13)
         | op.operation_6(14, user_kwarg=15)
         # The rest are PNs (except `operation_8`)
         | op.operation_7
         | op.operation_8(16, user_kwarg=17)
         | op.operation_9
   )

   assert pipeline[pni] == 9801 # Good luck figuring that one out ;)

Miscellaneous
=============

__getitem__
------------

For this entire tutorial, PNs and pipeline expressions have been evaluated using ``__getitem__``. There is actually another way to do this. As we learned, the first node in a pipeline only receives ``fpn.PN_INPUT`` as a kwarg. Not only that, but it **must** receive that as a kwarg. The call that kicks off a PN/pipeline evaluation must give a single kwarg:``fpn.PN_INPUT``

Thus, we can actually evaluate a PN/pipeline expression this way:

.. code:: python

   some_pipe_node(**{fpn.PN_INPUT: pni})

Obviously, this approach is not very pretty, and it's quite a lot to type for the privilege of evaluation. Thus, the ``__getitem__`` syntactical sugar was introduced to make it so the user isn't required to unpack a single kwarg whenever they want to evaluate a pipeline.

.. note::
   ``__getitem__`` has special handling for when the key is ``None``. This will evaluate the PN/pipeline expression with a bare instance of ``fpn.PipeNodeInput``. If the user desires to evaluate their expression with the literal value ``None``, they must kwarg unpack like so: ``pn(**{fpn.PN_INPUT: None})``.

Common Mistakes
---------------

1. Placing a bare factory in pipeline (see: A Common Factory Mistake).
2. Calling a PN directly (with the exception of unpacking the single kwarg ``fpn.PN_INPUT``).
3. Partialing a method wrapped with ``fpn.pipe_node`` or ``fpn.pipe_node_factory``.
4. Using ``@classmethod`` or ``@staticmethod`` decorators instead of the special decorators designed for working with classmethods/staticmethods.
5. Decorating a function with ``fpn.pipe_node`` whose signature expects args/kwargs outside either those bound from the pipeline, or ``**kwargs``.

Broadcasting
------------

A feature of ``fpn.pipe_node_factory`` is how it handles args/kwargs that are themselves PNs. For these types of arguments, it will evaluate them as isolated PNs with ``fpn.PN_INPUT`` forwarded, and then use the evaluated value in place of that PN. (This is referred to as broadcasting).

Example:

.. code:: python
   :class: copy-button

   @fpn.pipe_node_factory()
   def add_divide_exponentiate(*args, divide_by, to_power):
      return (sum(args) / divide_by) ** to_power

   @fpn.pipe_node(fpn.PN_INPUT)
   def multiply_input_by_2(pni):
      return pni * 2

   @fpn.pipe_node(fpn.PN_INPUT)
   def add_3_to_pni(pni):
      return pni + 3

   @fpn.pipe_node(fpn.PN_INPUT)
   def forward_pni(pni):
      return pni

   pipeline = add_divide_exponentiate(
      multiply_input_by_2,
      -4,
      forward_pni,
      divide_by=25,
      to_power=add_3_to_pni,
   )

   assert pipeline[12] == ((12 * 2 - 4 + 12) / 25) ** (12 + 3)

As we can see, when factories are given PNs as args/kwargs, they are evaluated with the ``fpn.PN_INPUT`` given to the original PN/expression being evaluated.

Arithmetic
----------

A helpful feature of PNs, is the ability to perform arithmetic operations on the pipeline during creation. Supported operators are:

- Unary: ``-``, ``~``, and ``abs()``
- Binary: ``+``, ``-``, ``*``, ``/``, ``**``, ``==``, ``!=``, ``>``, ``<``, ``<=``, and ``>=``

.. code:: python
   :class: copy-button

   @fpn.pipe_node(fpn.PN_INPUT)
   def get_pni(pni):
      return pni

   @fpn.pipe_node_factory(fpn.PREDECESSOR_RETURN)
   def mul(prev, val):
      return prev*val

   expr = ((get_pni + abs(-get_pni | mul(-0.9))) | mul(17) - 6 / get_pni) ** 23

   assert expr[12] == ((12 + abs(-12 * -0.9)) * 17 - 6 / 12) ** 23
