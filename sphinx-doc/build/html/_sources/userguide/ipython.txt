.. currentmodule:: pylada
.. _ipython_ug:

IPython high-throughput interface
*********************************

IPython_ is an ingenious combination of a bash-like terminal with a python
shell.  It can be used for both bash related affairs such as copying files
around creating directories, and for actual python programming. In fact, the
two can be combined to create a truly powerfull shell. 

Pylada puts this tool to good use by providing a command-line approach to
manipulate :ref:`job-folders <jobfolder_ug>`, launch actual calculations, and
collect the result.  When used in conjunction with python plotting libraries,
e.g. matplotlib_, it can provide rapid turnaround from conceptualization to
result analysis.

.. note:: 

   You may want to browse through the description of :ref:`job-folders
   <jobfolder_ug>` before reading this section.


.. warning:: 

   If you have not set-up Pylada to run :ref:`multiple mpi programs in parallel
   <configuration_mpi_ug>` yet, please add the one liner::

     do_multiple_mpi_programs = False 

   in your ~/.pylada file. Don't forget to remove it when time comes to run Pylada
   for real.

.. _ipython_prep_ug:

Prep
====

Pylada's IPython interface revolves around :ref:`job-folders <jobfolder_ug>`. 
In order to explore its features, we first need to create job-folders,
preferably some which do not involve heavy calculations. Please copy the
following to a file in the directory from which IPython_ is launched. In the
following, it will be refered to as "dummy.py".

.. literalinclude:: dummy.py
   :lines: 1-30, 32-

The above defines three functions: a dummy functional, an extraction function
capable of retrieving the results of the functional from disk, and a function
to create a simple job-folder. In real life, the functional could be a
:py:class:`~vasp.functional.Vasp` object. The extraction function would then be
:py:class:`~vasp.extract.Extract`. And the folder-creation function would depend
on the actual research project.

.. _ipython_manip_ug:

Manipulating job-folders
========================

Creating a job-folder
~~~~~~~~~~~~~~~~~~~~~

The job-folder could be created as described :ref:`here <jobfolder_ug>`.
However, it is easier -- and safer -- to create a script where a job-folder
creation function resides. If you have performed the step described :ref:`above
<ipython_prep_ug>`, and assuming that the resulting file is called dummy.py,
then a simple job-folder can be created from the ipython interface with:

>>> import dummy
>>> rootfolder = dummy.create_jobs()

.. tip:: 

   We need do ``import`` here because the functional is defined in the script
   itself. However, when using a functional defined somewhere else -- such as
   any Pylada functional -- it is easier for debugging purposes to do:

   >>> run -i dummy.py

   Each time the line above is called, the script is executed anew. When doing
   ``import``, it is executed only the `first time`_. 

.. _ipython_explore_ug:

Saving and Loading a job-folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this point we have job-folder stored in memory in a python variable. If you
were to exit ipython, the job-folder would be lost for ever and ever. 

>>> %savefolder dummy.dict rootfolder

The next time ipython is entered, the job-folder can be loaded from disk with: 

>>> %explore dummy.dict

Once a folder has been `explored` from disk, ``savefolder`` can be called
without arguments. 

The percent(%) sign indicates that these commands are ipython
`magic-functions`_. The percent can be obviated using `%automagic`_. To get
more information about what Pylada magic functions do, call them with "--help". 

.. tip::
   
   The current job-folder and the current job-folder path are stored in
   ``pylada.interactive.jobfolder`` and ``pylada.interactive.jobfolder_path``.
   In practice, accessing those directly is rarely needed.


Listing job-folders
~~~~~~~~~~~~~~~~~~~

The *executable* content of the current job-folder (the one loaded `via`
:ref:`%explore <ipython_explore_ug>`) can be examined with:

>>> %listfolders all
/GaAs/
/diamond/
/diamond/alloy/

This prints out the *executable* jobs. It can also be used to examine the
content of specific subfolders.

>>> %listfolders diamond/*
/diamond/
/diamond/alloy/

The syntax is the same as for the bash command-line. When given an argument
other than "all", `%listfolders` list all subfolders, even those which are not
executable. In practice, it works like "ls -d". 

.. _ipython_goto_ug:

Navigating the job-folders
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``%goto`` command reproduces the functionality of the "cd" unix command.

>>> %goto diamond
>>> %listfolders 
alloy
>>> %goto
Current position in job folder: /diamond/
Filename of job-folder:  /somedirectory/root.dict
>>> %goto ..
Current position in job folder: /
Filename of job-folder:  /somedirectory/root.dict

When called with argument, ``%goto`` prints the current location within the
job-folders. After the calculations are performed, directories will likely
exists with the same name as the subfolders. In that case, ``%goto`` also
changes the working directory to reflect the position in the subfolders.

Examining the executable content of a jobfolder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is always possible to change the executable data of a job-folder, whether
the functional or its parameters. To do this, we must first navigate to the
specific subfolder of interest, and then use the command ``%showme``.

>>> %goto diamond
>>> %showme functional

The command will launch an editor with the functional. Anything can be done at
this point. Once you're satisfied with then changes, save and close the file.
The file is then executed, and whatever functional is at the end of the script
will come and replace the functional in this subfolder. The functional is still
required to be an pickleable calleable. If any error is encountered, during the
execution of the script, then the original functional remains.

.. tip:: 

   Check the ipython documentation to select your favorite editor.
   Alternatively, set the EDITOR environment variable in your shell.

The parameters of the script can be edited the same way:

>>> %showme structure

This time, a scripts pops up with code defining the structure.
Finally, parameters can be added or removed by doing:

>>> %showme params

The complete dictionary, called params, appears for editing. Whatever
``params`` contains at the end of script will become the parameters passed on
to the functional when the job is eventually launched.

.. warning::
   Not all python objects can be translated back into python code. If giberrish
   pops up, then please turn to the next section.

.. _ipython_jobparams_ug:

Simultaneously examining/modify parameters for many jobs at a time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is likely that a whole group of calculations will share parameters in
common, and that these parameters need be the same. It is possible to examine
and modify any computational parameter for any number of jobs simultaneously:

>>> jobparams.structure.name
{
  '/GaAs/':          'Zinc-Blende',
  '/diamond/':       'Zinc-Blende',
  '/diamond/alloy/': 'Zinc-Blende',
}
>>> jobparams.structure.name = 'hello'
>>> jobparams.structure.name
{
  '/GaAs/':          'hello',
  '/diamond/':       'hello',
  '/diamond/alloy/': 'hello',
}

Note that there is no "%" sign in front of ``jobparams``.  Two things are
happening here. First, we can see that all jobs which contain
``structure.name`` are displayed. Secondly, we see that parameters can be
strung together: ``structure`` is an actual parameter for a calculation.
However, we are accessing and setting here an attribute which is owned by
``structure``. The functional and its attributes (and their attributes, and so
on) can be accessed in the same way. If a specific parameter does not exist for
any specific job-folder, then that job-folder is ignored. 

.. tip:: 

   It is only possible to *modify* attributes, as opposed to add new
   attributes. (Actually, the cake is lie. Experts hackers may want to
   check-out :py:attr:`JobParams.only_existing
   <jobfolder.manipulator.JobParams.only_existing>` and
   :py:data:`pylada.jobparams_only_existing`.)

Finally, it is possible to focus on a specific sub-set of jobfolders. By
default the syntax is that of a unix-shell. However, the syntax can be switched
to regular exppressions `via` the Pylada parameter :py:data:`pylada.unix_re`. Only
the former syntax is illustrated here:

>>> jobparams['*/alloy'].structure.name
'hello'

Note that in the last example, the name is returned directly, as opposed to a
dictionary as in previous examples. Indeed, there is only one job-folder
which corresponds to "\*/alloy". In that case, the object is returned as is,
rather than wrapped in a dictionary. This behavior can be turned on and off
using the parameters :py:data:`jobparams_naked_end` and
:py:attr:`JobParams.naked_end
<pylada.jobfolder.manipulator.JobParams.naked_end>`. The unix shell-like syntax
can be either absolute paths, when preceded with '/', or relative. In that last
case, they are relative to the current position in the job-folder, as changed
by :ref:`%goto <ipython_goto_ug>`. 

In most cases (see below) the return from using ``jobparams.something`` is a
:py:class:`kind of dictionary <pylada.jobfolder.forwarding_dict.ForwardingDict>`.
It can be iterated over like any other dictionary:

>>> for key, value in jobparams['diamond/*'].structure.name.iteritems():
...   print key, value
/diamond/ hello
/diamond/alloy/ hello

.. currentmodule:: pylada.jobfolder.manipulator

The other available iteration methods are :py:meth:`~JobParams.iterkeys`,
:py:meth:`~JobParams.keys`, :py:meth:`~JobParams.itervalues`,
:py:meth:`~JobParams.values`, :py:meth:`~JobParams.items`.

.. currentmodule:: pylada

.. _ipython_launch_ug:

Launching calculations
======================

Turning job-folders on and off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using ``jobparams``, it is possible to turn job-folders on and off:

>>> jobparams['diamond/alloy'].onoff = 'off'
>>> jobparams.onoff
{
  '/GaAs/':          'on',
  '/diamond/':       'on',
  '/diamond/alloy/': 'off',
}

When "off", a job-folder is ignored by ``jobparams`` (and ``collect``,
described below). Furthermore, it will not be executed. The only way to access
it again is to turn it back on. Groups of calculations can be turned on and off
using the unix shell-like syntax previously.


Submitting job-folder calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once job-folders are ready, it takes all of one line to launch the calculations:

>>> %launch scattered

This will create one pbs/slurm job per executable job-folder. A number of
options are possible to select the number of processors, the account or queue,
the walltime, etc. To examine them, do ``%launch scattered --help``. Most
default values should be contained in :py:data:`pylada.default_pbs`. The number
of processors is by default equal to the even number closest to the number of
atoms in the structure (apparently, this is a recommended VASP default). The
number of processes can be given both as an integer, or as function which takes
a job-folder as the only argument, and returns an integer.

There is currently only one other way of launching job-folders, ``%launch
interactive``. This will execute on job after another wherever the IPython_
instance is launched. For practical purposes, this is likely the best way to
launch the job-folder created earlier! Once that is done, you will find new
directories with names similar to those of the job-folders. Try navigating with
:ref:`%goto <ipython_goto_ug>`, you will find that both directories and
job-folders are changed simultaneously.

.. note::

  For ``%launch scattered`` to work, it is first necessary to setup Pylada to
  work with the :ref:`ressource manager <configuration_pbs_ug>` and with
  :ref:`MPI <configuration_mpi_ug>`.

.. _ipython_collect_ug:

Collecting results
==================

The first thing one wants to know from calculations is whether they ran:

>>> collect.success
{
  '/GaAs/':          True,
  '/diamond/':       True,
  '/diamond/alloy/': True,
}

Our dummy functional is too simple to fail... However, if you delete any given
calculation directory, and try it again, you will find some false results.
Beware that some collected results are cached so they can be retrieved faster
the second time around, so redoing ``%explore some.dict`` might be necessary.

.. warning:: Success means that the calculations ran to completion. It does not
             mean that they are not garbage.

Results from the calculation can be retrieved in much the same way as
parameters were examined. This time, however, we use an object called
``collect`` (still without preceding "%" sign). Assuming the job-folders
created earlier were launched, the random energies created by our fake
functional could be retrieved as in:

>>> collect.energy
{
  '/GaAs/':          0.8231,
  '/diamond/':       0.5452,
  '/diamond/alloy/': 0.0312,
}

What exactly can be collected this way will depend on the actual calculation.
The easiest way to examine what's available it to hit ``collect.[TAB]``.
The collected results can be iterated over, focussed to a few relevant
calculations, exactly as was done with :ref:`jobparams <ipython_jobparams_ug>`.
The advantage is that further job-folders can be easyly constructed which
take the calculations a bit further. For instance, we have created job-folders
which minimize spin-polarized crystal structures. Then a second-wave of
job-folders would be created from the resulting relaxed crystal structures to
examine different possible magnetic orders.

On other nice feature is to use ``collect``  in conjunction with matplotlib_ to
plot results, as described :ref:`here <vasp_massextract_ug>`. In practive,
``collect`` interfaces with a lot more properties than just the energy, as
displayed here. 

.. _IPython: http://ipython.org/
.. _first time: http://ipython.org/ipython-doc/stable/config/extensions/autoreload.html
.. _matplotlib: http://matplotlib.sourceforge.net/
.. _magic-functions: http://ipython.org/ipython-doc/dev/interactive/tutorial.html#magic-functions
.. _%automagic: http://ipython.org/ipython-doc/dev/interactive/reference.html#magic-command-system
