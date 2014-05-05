
schedMain.py: General task scheduler
=====================================

The schedMain package is a general scheduler designed for
tasks running on a Linux or similar operating system.

SchedMain maintains a **work list** of tasks to be executed.
Each item in the work list represents a specific task to be
run in a specific directory.  When schedMain starts it reads
an initial work list from the file initWork.  Each line of 
initWork contains the name of a task and the directory to run in.
For example initWork might contain::

  alpha.py  aaDir

This represents the task alpha.py to be run in directory aaDir.

Using the above example of a work list containing alpha.py with
directory aaDir ...

Before task alpha.py can run in aaDir,
schedMain checks for a file ``aaDir/alpha.preWork``.
If the preWork file exists it contains a list
of tasks that must complete successfully before alpha.py can start
in aaDir.  The preWork file has the same format as the initWork file.

After the prerequisites complete schedMain starts alpha.py.
Eventually alpha.py signals its completion by writing a tiny file
``aaDir/alpha.status.ok``, indicating OK completion.
Every task must write either a ``x.status.ok`` or ``x.status.error``
file on completion, to let schedMain know it's done.
If a task completes without writing either file, schedMain
assumes it ended badly and will write an ``x.status.error``
for it.

After task alpha.py completes schedMain removes it from the work list.
If the task was successful (wrote aaDir/alpha.status.ok),
schedMain checks for a file ``aaDir/alpha.postOkWork``.
If the postOkWork file exists, it contains a list
new tasks to be added to schedMain's work list.
The postOkWork file has the same format as initWork.

Now would be a good time to check out
the static example, :ref:`example.static`,





:mod:`schedMain` Package
--------------------------

.. automodule:: schedMain.__init__
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`schedMain` Module
-----------------------

.. automodule:: schedMain.schedMain
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`schedMisc` Module
-----------------------

.. automodule:: schedMain.schedMisc
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`taskClass` Module
-----------------------

.. automodule:: schedMain.taskClass
    :members:
    :undoc-members:
    :show-inheritance:

