###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
# 
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
#  crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
# 
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
# 
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
# 
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################

from .process import Process
class IteratorProcess(Process):
  """ Executes an iteration function in child process. 
  
      An iterator process is a *meta*-process which runs other processes
      sequentially. It is one which needs iterating over, as shown more
      explicitely below. Its interface is fairly similar to other processes.

      .. code-block:: python

        program = ProgramProcess(generator, outdir=dir, **kwargs)
        program.start(comm)
        try: program.wait()
        except Fail:
          # do something
        
      The main difference is in ``generator``: it is kind of *functional* which
      will make more than one call to an external program. For instance, it
      could be several calls to VASP, a first call to relax the strain,
      followed by a static calculation. We want a process interface which hides
      those details (i.e. the sequence of calls to one or more external
      programs) from the owner of the process: the owner only wants to know
      that the calculation is still on-going, not which particular step it is
      at.

      To do this, we use a generator_, i.e. an object which can be used in a
      loop. It should yield_ one of two kinds of objects: (i) an extractor.
      such as :py:class:`pylada.vasp.extract.Extract`, or an instance derived
      from :py:class:`~pylada.vasp.process.Process`. In the former case, the
      :py:class:`IteratorProcess` simply keeps looping. In the latter case, the
      yielded process object is started and control is relinquished to the
      owner of the :py:class:`IteratorProcess` instance. When the looping is
      done, the final extraction object that was ever yielded is returned.

      This approach can be put into the following (pseudo) code:

      .. code-block:: python

        for process in generator(**kwargs):
          if hasattr(program, 'success'): 
            result = process
            # go to next iteration
            continue
          else:
            # start process obtained in this iteration
            process.start(self._comm)
            # relinquish control to owner of self
        # return the final extraction object
        return result

      A ``generator`` which implements two calls to an external program
      ``EXTERNAL`` would look something like this:

      .. code-block:: python

        def generator(...):
          # PERFORM FIRST CALCULATION
          # check for success, e.g. pre-existing calculation
          if Extract(outdir0).success: 
            yield Extract(outdir0)
          # otherwise yield a program process wich calls EXTERNAL
          else:
            yield ProgramProcess(EXTERNAL)
            # check for success of this calculation.
            # if an error occured, handle it or throw an error
            if not Extract(outdir0).success: raise Exception()
              

          # PERFORM SECOND CALCULATION
          # check for success, e.g. pre-existing calculation
          if Extract(outdir1).success: 
            yield Extract(outdir1)
          # otherwise yield a program process wich calls EXTERNAL
          else: yield ProgramProcess(EXTERNAL)

          # yield the final extraction object. 
          # this is usefull for things other than process management.
          yield ExtractFinal(outdir)

      To implement it, we need an extraction class, such as
      :py:class:`pylada.crystal.extract.Extract`, capable of checking in a
      directory whether a successfull calculation already exists. If it does,
      the generator should yield the extraction object. If it doesn't, it
      should yield some :py:class:`~pylada.process.process.Process` instance
      capable of starting the external program of interest. 

      The processes yielded by the input generator could be anything. It could
      be, for instance, another instance of :py:class:`IteratorProcess`, or
      simply a bunch of :py:class:`~pylada.process.program.ProgramProcess`, or
      something more comples, as long as it presents the interface defined by
      :py:class:`~pylada.process.process.Process`.

      .. note::

        It would be possible, of course, to simply create a python program which
        does all the calls to the external programs itself. Then we wouldn't
        have to deal with generators and loops explicitely. However, such an
        approach would start one more child python program than strictly
        necessarily, and hence would be a bit heavier. Nevertheless, since that
        approach is somewhat simpler, it has been implemented in
        :py:class:`~process.callprocess.CallProcess`.

      .. seealso::
       
        :py:class:`~process.callprocess.CallProcess`,
        :py:class:`~process.jobfolder.JobFolderProcess`,
        :py:class:`~.pool.PoolProcess`.

      .. _generator: http://docs.python.org/tutorial/classes.html#generators
      .. _yield: http://docs.python.org/reference/simple_stmts.html#yield
  """
  def __init__(self, functional, outdir, maxtrials=1, **kwargs):
    """ Initializes a process.
    
        :param functional:
          A generator which yields processes and/or extraction objects.
        :param str outdir: 
          Path where the processes should be executed.
        :param int maxtrials:
          Maximum number of times to try re-launching each process upon
          failure. 
        :param kwargs:
          Keyword arguments to the generator should be given here, as keyword
          arguments to :py:class:IteratorProcess`.
    """
    from ..misc import RelativePath
    super(IteratorProcess, self).__init__(maxtrials)
    self.functional = functional
    """ Iterable to execute. """
    self.outdir = RelativePath(outdir).path
    """ Execution directory of the folder. """
    self._iterator = None
    """ Current iterator. """
    self.params = kwargs.copy()
    """ Extra parameters to pass on to iterator. """

  def poll(self):
    from . import Fail

    # checks whether program was already started or not.
    if super(IteratorProcess, self).poll(): return True

    # check if we have currently running process.
    # catch StopIteration exception signaling that process finished.
    found_error = None
    try:
      if self.process.poll() == False: return False
    except Fail as failure: found_error = failure
    try: self.process._cleanup()
    finally: self.process = None

    # if stopped on or before previous job, then we have a retrial.
    if found_error is not None:
      self.nberrors += 1
      if self.nberrors >= self.maxtrials:
        self._cleanup()
        raise found_error if isinstance(found_error, Fail) \
              else Fail(str(found_error))

    # At this point, go to next iteration.
    process = self._get_process()
    if process is not None:
      self._next(process)
      return False

    self._cleanup()
    return True
  poll.__doc__ = Process.poll.__doc__


  def _get_process(self):
    """ Finds next available iteration. """
    from . import Fail
    # first creates iterator, depending on input type.
    if self._iterator is None:
      iterator = self.functional.iter if hasattr(self.functional, 'iter')\
                 else self.functional
      self._iterator = iterator( comm=self._comm,
                                 outdir=self.outdir, 
                                 **self.params)
    try:
      result = self._iterator.next()
      while hasattr(result, 'success'): 
        result = self._iterator.next()
      return result
    except StopIteration: return None
    except Exception as e:
     import sys, traceback
     exc_type, exc_value, exc_traceback = sys.exc_info()
     tb = traceback.format_tb(exc_traceback)
     raise Fail( '{0}: {1}\nError occured in {3} with '                        \
                 'the following traceback:\n{2}'                               \
                 .format(type(e), e, '\n'.join(tb), self.outdir) )

  def _next(self, process=None):
    """ Launches next process. """
    # start next process.
    self.process = process if process is not None else self._get_process()
    if self.process is None: return True
    self.process.start(self._comm.lend('all')) 
    return False

  def start(self, comm):
    if super(IteratorProcess, self).start(comm): return True
    self._next()
    return False
  start.__doc__ = Process.start.__doc__

  def wait(self):
    """ Waits for process to end, then cleanup. """
    if super(IteratorProcess, self).wait(): return True
    while not self.poll(): self.process.wait()
    self._cleanup()
    return False
