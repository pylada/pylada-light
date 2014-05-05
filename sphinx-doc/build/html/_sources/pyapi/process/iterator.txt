Sequential execution of a generator of processes
************************************************

.. currentmodule:: pylada.process.iterator
.. moduleauthor:: Mayeul d'Avezac
.. autoclass:: IteratorProcess
   :show-inheritance:
   :members:

   .. automethod:: __init__

   .. automethod:: _cleanup

   .. attribute:: process

      Holds currently running process. 

      This would be the latest process yielded by the input generator
      :py:attr:`functional`.
