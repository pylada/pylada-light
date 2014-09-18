.. currentmodule:: pylada.jobfolder
.. _jobfolder_ug: 

Organized high-throughput calculations: job-folders
***************************************************

Pylada provides tools to organize high-throughput calculations in a systematic
manner.  The whole high-throughput experience revolves around **job-folders**.
These are convenient ways of organizing actual calculations. They can be though
of as folders on a file system, or directories in unix parlance, each one
dedicated to running a single actual calculation (eg launching :ref:`VASP
<vasp_ug>` once). The added benefits beyond creating the same file-structure
with bash are:

 1. the ability to create a tree of folders/calculations using the power of the
    python programming language. No more copy-pasting files and unintelligible
    bash scripts!
 2. the ability to launch all folders simultaneously
 3. the ability to collect the results across all folders simultaneously, all
    within python, and with all of python's goodies. E.g. no more copy-pasting
    into excel by hand. Just do the summing, and multiplying, and graphing
    there and then.


Actually, there are a lot more benefits. Having everything - from input to
output - within the same modern and efficient programming language means there
is no limit to what can be achieved.

The following describes how job-folders are created. The fun bits, 
launching jobs, collecting results, manipulating all job-folders
simultaneously, can be found in the next section. Indeed, all of these are
intrinsically linked to the Pylada's IPython interface.

Prep
~~~~
First off, we will need a functional. Rather that use something heavy, like
VASP, we will use a dummy functional which does pretty much nothing... Please
copy the following into a file, any file, which I recommend to call dummy.py.
Putting it into a file is important because we will want python to be able to
refer to it later on.

.. literalinclude:: dummy.py
   :lines: 18-28, 31

This functional takes a few arguments, amongst which an output directory, and
writes a file to disk. That's pretty much it.


Creating and accessing job-folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Job-folders can be created with two simple lines of codes:

  >>> from pylada.jobfolder import JobFolder
  >>> root = JobFolder()

To add further job-folders, one can do:

  >>> jobA = root / 'jobA'
  >>> jobB = root / 'another' / 'jobB'
  >>> jobBprime = root / 'another' / 'jobB' / 'prime'

As you can, see job-folders can be given any structure that on-disk directories
can. What is more, a job-folder can access other job-folders with the same kind
of syntax that one would use (on unices) to access other directories:

  >>> jobA['/'] is root
  True
  >>> jobA['../another/jobB'] is jobB
  True
  >>> jobB['prime'] is jobBprime
  True
  >>> jobBprime['../../'] is not jobB
  True
  >>> root['..']
  KeyError: 'Cannot go below root level.'


Furthermore, job-folders know what they are:

  >>> jobA.name
  '/jobA/'
  
What their parents are:

  >>> jobB.parent.name
  '/another/'

And what the root is:

  >>> jobBprime.root is root
  True
  >>> jobBprime.root.name
  '/'

They also know what they contain:

  >>> 'prime' in jobB
  True
  >>> '/jobA' in jobBprime
  True

Making a job-folder executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The whole point of a job-folder is to create an architecture for calculations.
Each job-folder can contain at most a single calculation. A calculation is
setup by passing to the job-folder a function and the parameters for calling it.

  >>> from pylada.crystal.binary import zinc_blende
  >>> from dummy import functional
  >>>
  >>> jobA.functional = functional
  >>> jobA.params['structure'] = zinc_blende()
  >>> jobA.params['value'] = 5

In the above, the function ``functional`` from the dummy module created
previously is imported into the namespace. The special attribute
:py:attr:`job.functional <jobfolder.Jobfolder.functional>` is
set to ``functional``. Two arguments, ``structure`` and ``value``, are
specified by adding the to the dictionary :py:attr:`job.params
<jobfolder.Jobfolder.params>`. Please note that the third
line does not contain parenthesis: this is not a function call, it merely saves
a reference to the function with the object of calling it later. 'C' aficionados
should think a saving a pointer to a function.

.. warning:: The reference to ``functional`` is deepcopied_: the instance that
   is saved to jod-folder is *not* necessarily the one that was passed to i.
   On the other hand, the parameters (``jobA.params``) are held by reference
   rather than by value.


.. tip:: To force a job-folder to hold a functional by reference rather than by
   value, do:

   >>> jobA._functional = functional

The parameters  in ``job.params`` should be pickleable_ so that the folder can
be saved to disk later.  :py:attr:`~jobfolder.Jobfolder.functional` must be a
pickleable_ callable_. Setting :py:attr:`~jobfolder.Jobfolder.functional` to
something else will immediately fail. In practice, this means it can be a
function or a callable class, as long as that function or class is imported
from a module. It cannot be defined in `__main__`__, e.g. the script that you
run to create the job-folders:

>>> run -i jobscript.py # functional must defined outside jobscript.py.

However, if jobscript is imported as a module, and the job-folders are
created via a function, then ``functional`` can be defined inside
jobscript.py:

>>> import jobscript
>>> newjobs = jobscript.create_my_jobfolders() # functional can be defined in jobscript.py


These complications are due to the way python pickles_ data.  And pickling we
need to save job-folders to disk.  The functional is called with the parameters
passed to the folder as keyword arguments:

>>> jobA.compute(outdir=jobA.name[1:])

is exactly equivalent to:

>>> functional(structure=zinc_blende(), value=5, outdir='jobA')

Note that we have passed an extra argument ``outdir``, which is the output
directory. It is customary to set it to the name of the job (minus the leading /).
Any one of the two previous commands will create a "JobA" sub-directory in the
current directory.

.. tip::

   Executable olders can be iterated the same way dictionaries can, with
   :py:meth:`~jobfolder.JobFolder.keys`,
   :py:meth:`~jobfolder.JobFolder.iterkeys`, 
   :py:meth:`~jobfolder.JobFolder.values`, 
   :py:meth:`~jobfolder.JobFolder.itervalues`, 
   :py:meth:`~jobfolder.JobFolder.items`, 
   :py:meth:`~jobfolder.JobFolder.iteritems`.

Saving and loading folders
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :ref:`IPython interface <ipython_ug>` provides better ways to both.
However, it is still possible to load and save job-folders to disk from a
script:

>>> from pylada.jobfolder import load, save
>>> save(root, 'root.dict') # saves to file
>>> root = load('root.dict') # loads from file

The file format is a pickle_. It is not meant for human eyes. However, it can
be transferred from one computer to the next. The parameters ``job.params``
should be pickleable_, as well as the functional, for this to work.
The advantage of using these two functions is that they take care of locking
access to file on-disk before reading or writing to it. This way, multiple
processes can access the file without fear of getting into one another's way.

.. tip:: 

   If either load or save takes for ever, check whether the lock-directory
   ".filename-pylada_lockdir" exists. If you are *sure* that no other process
   exists which is trying to access the file on disk, then you can delete the
   lock-directory and try saving/loading again. Alternatively, a timeout
   argument can be provided to raise an exception if the file cannot be locked.

.. __: http://docs.python.org/library/__main__.html
.. _pickleable: http://docs.python.org/library/pickle.html#what-can-be-pickled-and-unpickled
.. _callable: http://docs.python.org/reference/datamodel.html#emulating-callable-objects
.. _pickles: http://docs.python.org/library/pickle.html
.. _pickle: pickles_
.. _deepcopied: http://docs.python.org/library/copy.html#copy.deepcopy
