.. _vasp_example_ug:

Example: Epitaxial relaxation method
====================================

Absent dislocations, a film grown epitaxially on a substrate adopts the
in-plane lattice parameters of that substrate. It is possible to model a
sufficiently thick film as a bulk material on a virtual substrate. The trick is
simply to lock the in-plane parameters while allowing the out-plane direction
to relax.

VASP_ does not allow for this kind of relaxation directly. However, using Pylada
when can simply design a method which will magically perform the relaxation.
Below, a method based on the secant_ method is presented:

  1. Find interval for which stress changes sign.
  2. Bisect interval until required accuracy is obtained.

Admittedly, if one were smart, one could design a faster optimization method.
Or one could use the optimization methods provided by scipy_ with a
:py:class:`~functional.Vasp` object, and be done with it, say this one_.

First, we need a function capable of creating a strain matrix for the
particular kind we have in mind, and apply it to the structure. The strain
matrix can be obtained simply as the outer product of the epitaxial direction
with itself.

.. literalinclude:: epirelax.py
   :lines: 1-12

The next step is to create a function which will compute the total energy of a
strained structure while relaxing internal degrees of freedom. 
VASP_ is invoked in a separate sub-directory.

.. literalinclude:: epirelax.py
   :lines: 14-17

The third component is a function to return the stress component in the
epitaxial direction.

.. literalinclude:: epirelax.py
   :lines: 19-22

Finally, we have all the pieces to put the bisecting algorithm together: 

.. literalinclude:: epirelax.py
   :lines: 24-25, 68-94
   :linenos:

Lines 4 through 16 correspond to (1) above. Starting from the initial input
structure, we change the strain until an interval is found for which the stress
component changes direction. The structure with minimum total energy will
necessarily be found within this interval. Line 14 makes sure the interval is
kept as small as possible.

We then move to the second part (lines 19-25) of the algorithm: bisecting the
interval until the requisite convergence is achieved. Finally, one last static
calculation is performed before returning. That's probably an overkill, but it
should be fairly cheap since we restart from the charge density and
wavefunctions of a previous calculations.

The full listing of the code is given below. There are some extra bits to it,
namely a docstring and some sanity checks on the function's input parameter.

.. literalinclude:: epirelax.py

.. _one: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html
.. _secant: http://en.wikipedia.org/wiki/Secant_method
