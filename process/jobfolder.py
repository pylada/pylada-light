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
class JobFolderProcess(Process):
  """ Executes folder in child processes.
  
      Expects a jobfolder on input. Executable job-folders are launched in
      parallel, with up to :py:attr:`~JobFolderProcess.nbpools` running
      instances. Each instance is allocated an equal number of processors.

      If a folder does not execute correctly, then the failure code or
      exception is stored in :py:attr:`errors` until all folders have been
      executed. Only then is a :py:exc:`~pylada.process.Fail` exception raised. 

      .. note:: 

        The executable folders are determined when the process is first
        created. To modify :py:attr:`jobfolder`, one should call
        :py:meth:`update`.

      .. seealso:: :py:class:`~pylada.process.pool.PoolProcess`
  """
  def __init__( self, jobfolder, outdir, maxtrials=1, nbpools=1,
                keepalive=False, **kwargs ):
    """ Initializes a process.
    
        :param jobfolder:
          Jobfolder for which executable folders should be launched.
          The name of the folders to launch are determined which
          :py:meth:`__init__` is acalled. If ``jobfolder`` changes, then one
          should call :py:meth:`update`.
        :type jobfolder: :py:class:`~pylada.jobfolder.jobfolder.JobFolder` 
        :param str outdir: 
          Path where the python child process should be executed.
        :param int nbpools:
          Maximum number of executable folders to run in parallel. The
          processors will splitted into *n* of approximately equal length, where
          *n* is ``nbpool`` or the remaining number of executable jobs, which
          ever is smallest.
        :param bool keepalive:
           Whether to relinquish communicator once jobs are completed.  If
           True, the communicator is not relinquished. The jobfolder can be
           :py:meth:`updated <update>` and new jobs started. To finally
           relinquish the communicator, :py:attr:`keepalive` should be set to
           False.  Both :py:meth:`kill` and :py:meth:`terminate` ignore this
           attribute and relinquish the communicator. However, since both side
           effects, this may not be the best way to do so.
        :param int maxtrials:
          Maximum number of times to try re-launching each process upon
          failure. 
        :param kwargs:
          Keyword arguments to the functionals in the executable folders. These
          arguments will be applied indiscriminately to all folders.
    """
    from ..misc import RelativePath
    super(JobFolderProcess, self).__init__(maxtrials)

    self.jobfolder = jobfolder
    """ Jobfolder for which executable folders should be launched.

        The name of the folders to launch are determined which
        :py:meth:`__init__` is acalled. If ``jobfolder`` changes, then one
        should call :py:meth:`update`.
    """
    self.outdir = RelativePath(outdir).path
    """ Execution directory of the folder. """
    self.process = []
    """ List of currently running processes. 
    
        Each item consists of an index into the job-folder,
        an instance derived from :py:class:`~pylada.process.process.Process`,
        e.g. :py:class:`~pylada.process.call.CallProcess`, and a communicator
        used by that process.
    """
    self.nbpools = nbpools
    """ Number of executable folders to launch in parallel. 

        The processors will splitted into *n* of approximately equal length,
        where *n* is ``nbpool`` or the remaining number of executable jobs,
        which ever is smallest.
    """
    self._finished = set()
    """ Set of finished runs. """
    self._torun = set()
    """ List of jobs to run. """
    for name, job in self.jobfolder.iteritems():
      if not job.is_tagged: self._torun.add(name)

    self.errors = {}
    """ Map between name of failed jobs and exception. """
    self.keepalive = keepalive
    """ Whether to relinquish communicator once jobs are completed. 
    
        If True, the communicator is not relinquished. The jobfolder can be
        :py:meth:`updated <update>` and new jobs started. To finally relinquish
        the communicator, :py:attr:`keepalive` should be set to False.  Both
        :py:meth:`kill` and :py:meth:`terminate` ignore this attribute and
        relinquish the communicator. However, since both side effects, this may
        not be the best way to do so.
    """
    self.params = kwargs.copy()
    """ Keyword arguments to the functionals in the executable folders. 
    
        These arguments will be applied indiscriminately to all folders.
    """

  @property
  def nbjobsleft(self): 
    """ Number of jobs left. """
    return len(self._torun)

  def poll(self): 
    """ Polls current job. """
    from . import NotStarted
    from . import Fail

    if self.nbjobsleft == 0 and super(JobFolderProcess, self).poll():
      return True
    if not hasattr(self, '_comm'): raise NotStarted("Process was never started.")

    # weed out successfull and failed jobs.
    finished = []
    for i, (name, process) in enumerate(list(self.process)):
      try:
        if process.poll() == True: 
          self._finished.add(name)
          finished.append(i)
      except Exception as e:
        self.errors[name] = e
        finished.append(i)
    for i in sorted(finished)[::-1]:
      name, process = self.process.pop(i)
      process._cleanup()

    # returns True if nothing left to do.
    if len(self.process) == 0 and len(self._torun) == 0:
      self._cleanup()
      if len(self.errors) == 0: return True
      else: raise Fail(str(self.errors))

    # adds new jobs. 
    self._next()
    return False
 
  def _next(self):
    """ Adds more processes.
    
        This is the subroutine to overload in a derived class which would
        implement some sort of scheduling.
    """
    from os.path import join
    from ..error import IndexError
    from .call import CallProcess
    from .iterator import IteratorProcess

    # nothing else to do.
    if len(self._torun) == 0: return
    # cannot add more processes.
    if len(self.process) >= self.nbpools: return
    # no more machines to allocate...
    if self._comm['n'] == 0: return

    # split processes into local comms. Make sure we don't oversuscribe.
    njobs = min(self._comm['n'], self.nbpools - len(self.process))
    local_comms = self._comm.split(njobs)
    try: 
      # Loop until all requisite number of processes is created, 
      # or until run out of jobs, or until run out of comms. 
      while len(self.process) < self.nbpools \
            and len(self._torun) > 0         \
            and len(local_comms) > 0:
        name = self._torun.pop()
        # checks folder is still valid.
        if name not in self.jobfolder:
          raise IndexError("Job-folder {0} no longuer exists.".format(name))
        jobfolder = self.jobfolder[name]
        if not jobfolder.is_executable:
          raise IndexError("Job-folder {0} is no longuer executable.".format(name))
        # creates parameter dictionary. 
        params = jobfolder.params.copy()
        params.update(self.params)
        params['maxtrials'] = self.maxtrials
        # chooses between an iterator process and a call process.
        if hasattr(jobfolder.functional, 'iter'):
          process = IteratorProcess( jobfolder.functional,
                                     join(self.outdir, name), **params )
        else:
          process = CallProcess(self.functional, join(self.outdir, name), **params)
        # appends process and starts it.
        self.process.append((name, process))
        try: process.start(local_comms.pop())
        except Exception as e:
          self.errors[name] = e
          name, process = self.process.pop(-1)
          process._cleanup()
          raise
    except:
      self.terminate()
      raise
    finally:
      for comm in local_comms: comm.cleanup()

  def kill(self):
    """ Kills all currently running processes. 
    
        Relinquishes communicator, even if
        :py:attr:`~JobProcessFolder.keepalive` is True.
    """
    for name, process in self.process: process.kill()
    self.keepalive, oldkeepalive = False, self.keepalive
    try: self._cleanup()
    except:
      self.keepalive = oldkeepalive
      raise
    
  def terminate(self):
    """ Kills all currently running processes. 
    
        Relinquishes communicator, even if
        :py:attr:`~JobProcessFolder.keepalive` is True.
    """
    for name, process in self.process: process.terminate()
    self.keepalive, oldkeepalive = False, self.keepalive
    try: self._cleanup()
    except:
      self.keepalive = oldkeepalive
      raise

  @property 
  def done(self):
    """ True if job already finished. """
    return self.started and len(self.process) == 0
  @property
  def nbrunning_processes(self):
    """ Number of running processes. 

        For simple processes, this will be one or zero.
        For multitasking processes this may be something more.
    """
    return 0 if (not self.started) or len(self.process) == 0 else 1

  def wait(self, sleep=1):
    """ Waits for all job-folders to execute and finish. """
    from time import sleep as ossleep
    from . import NotStarted
    if self.nbjobsleft == 0 and super(JobFolderProcess, self).wait():
      return True
    if not hasattr(self, '_comm'): raise NotStarted("Process was never started.")
    while self.poll() == False: ossleep(sleep)
    return False

  def _cleanup(self):
    """ Cleans up after currently running processes. """
    try: 
      for name, process in self.process:
        try: process._cleanup()
        except: pass
    finally:
      self.process = []
      if hasattr(self, '_comm') and self.keepalive == False: 
        try: self._comm.cleanup()
        finally: del self._comm
   
  def start(self, comm):
    """ Start executing job-folders. """
    if super(JobFolderProcess, self).start(comm): return True
    self._next()
    return False

  def update(self, jobfolder, deleteold=False):
    """ Updates list of jobs.
    
        Adds jobfolders which are not in ``self.jobfolder`` but in the input.
        Updates jobs which are in ``self.jobfolder`` and input if not currently
        running.  Does nothing if are job is currently running.
        If ``deleteold`` is True, then removed finished jobs from job-folder.
    """
    running = set([n for n in self.process])
    for name, value in jobfolder.root.iteritems():
      if value.is_tagged: continue
      if name in running: continue
      elif name not in self.jobfolder.root:
        newjob = self.jobfolder.root / name
        newjob.functional = value.functional
        newjob.params.update(value.params)
        for key, value in value.__dict__.iteritems():
          if key in ['children', 'params', '_functional', 'parent']: continue
          setattr(self, key, value)
        self._torun.add(name)
      elif name not in self._finished:
        self.jobfolder.root[name] = value
    for name in self.jobfolder.root.iterkeys():
      if name in self._finished and deleteold: del self.jobfolder.root[name]
