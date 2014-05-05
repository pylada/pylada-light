==============
Process Module
==============

.. automodule:: pylada.process

  .. moduleauthor:: Mayeul d'Avezac

*************
Process Types
*************

.. toctree::
   :maxdepth: 1

   Abstract base-class <process/process>
   
   Execution of an external program <process/program>

   Sequential execution of a generator of processes <process/iterator>

   Execution of a callable <process/call>

   Parallel execution of a job-folder <process/jobfolder>

   Parallel execution of a job-folder with per-folder numbers of processors <process/pool>

**********
Exceptions
**********

.. autoexception:: ProcessError
   :show-inheritance:

.. autoexception:: Fail
   :show-inheritance:

.. autoexception:: AlreadyStarted
   :show-inheritance:

.. autoexception:: NotStarted
   :show-inheritance:

*********
Functions
*********

.. autofunction:: which
