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

from .jobfolder import JobFolderProcess
class PoolProcess(JobFolderProcess):
  """ Executes folder in child processes.
  
      Much as its base class,
      :py:class:`~pylada.process.jobfolder.JobFolderProcess`, this process
      specialization is intended to run jobs in a jobfolder in parallel [*]_.
      However, it allows to customize the number of processors dedicated to
      each job, rather than use the same number of processors for each job. 

      The customization is done *via* the function :py:attr:`processalloc`. It
      takes one argument, the executable jobfolder, and returns an integer
      signifying the requested number of processors.

      .. code-block:: python

        def processalloc(folder):
          return (len(folder.structure) // 2) * 2

        process = PoolProcess(jobfolder, outdir='here', processalloc=processalloc)
        process.start(comm)

        try: process.wait()
        except Fail: pass

      The interface is much the same as any other process. However, it takes as
      argument this :py:attr:`processalloc` function, on top of the jobfolder
      itself. In this case, each folder will be launched with approximately as
      many processors as there are atoms in the structure [*]_.

      Once it is launched, the :py:class:`PoolProcess` instance will attempt to
      run as many jobs as possible in parallel, until there it runs out of
      processors to allocate. Howe many processors, and which machines, is
      determined by the communicator passed to :py:meth:`start`. Each time an
      executable folder is finished [*]_, it tries again to pack jobs into the
      available processor pool. 

      .. note::
      
         Upon failure, :py:exc:`~pylada.process.Fail` is raised only
         once all the folders have been executed, not when the failure is
         detected.
      
      .. [*] Several job-folders are executed simultaneously, not
        withstanding the possibility that each of these is also executed in
        parallel *via* MPI.
      .. [*] Apparently, this is a pretty good rule-of-thumb for VASP
        calculations.
      .. [*] More, specifically, each time
         :py:meth:`~pylada.process.jobfolder.JobFolderProcess.poll` is called. 
  """
  def __init__( self, jobfolder, outdir, processalloc, maxtrials=1,
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
        :param processalloc:
          Function which determines how many processors each job requires.
          This is determined for each job when this instance is created. To
          change :py:attr:`~pylada.process.jobfolder.JobFolderProcess.jobfolder`,
          one should call :py:meth:`update`.
        :type processalloc:
          (:py:class:`~pylada.jobfolder.jobfolder.JobFolder`)->int
        :param bool keepalive:
           Whether to relinquish communicator once jobs are completed.  If
           True, the communicator is not relinquished. The jobfolder can be
           :py:meth:`updated <update>` and new jobs started. To finally
           relinquish the communicator,
           :py:attr:`~pylada.process.jobfolder.JobFolderProcess.keepalive`
           should be set to False.  Both
           :py:meth:`~pylada.process.jobfolder.JobFolderProcess.kill` and
           :py:meth:`~pylada.process.jobfolder.JobFolderProcess.terminate` ignore
           this attribute and relinquish the communicator. However, since both
           side effects, this may not be the best way to do so.
        :param int maxtrials:
          Maximum number of times to try re-launching each process upon
          failure. 
        :param kwargs:
          Keyword arguments to the functionals in the executable folders. These
          arguments will be applied indiscriminately to all folders.
    """
    super(PoolProcess, self).__init__( jobfolder, outdir, maxtrials,
                                       keepalive=keepalive,  **kwargs )
    del self.nbpools # not needed here.

    self.processalloc = processalloc
    """ Determines number of processors to allocate to each job.
    
        This is a function which takes a
        :py:class:`~pylada.jobfolder.jobfolder.JobFolder` instance and returns an
        integer.
    """
    self._alloc = {}
    """ Maps job vs rquested process allocation. """
    for name in self._torun:
      self._alloc[name] = self.processalloc(self.jobfolder[name])
    assert len(set(self._alloc.keys())) == len(self._alloc)


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
    # no more machines to allocate...
    if self._comm['n'] == 0: return
    # cannot add more processes.
    if isinstance(self.processalloc, int):
      return super(PoolProcess, self)._next()

    # split processes into local comms. Make sure we don't oversuscribe.
    jobs = self._getjobs()
    assert sum(self._alloc[u] for u in jobs) <= self._comm['n']
    try: 
      # Loop until all requisite number of processes is created, 
      # or until run out of jobs, or until run out of comms. 
      for name in jobs:
        self._torun = self._torun - set([name])
        nprocs = self._alloc.pop(name)
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
          process = IteratorProcess(jobfolder.functional, join(self.outdir, name), **params)
        else:
          process = CallProcess(self.functional, join(self.outdir, name), **params)
        # appends process and starts it.
        self.process.append((name, process))
        try: process.start(self._comm.lend(nprocs))
        except Exception as e:
          self.errors[name] = e
          name, process = self.process.pop(-1)
          process._cleanup()
    except:
      self.terminate()
      raise

  def _getjobs(self):
    """ List of jobs to run. """
    from operator import itemgetter
    N = self._comm['n']

    # creates list of possible jobs.
    availables = sorted( [(key, u) for key, u in self._alloc.iteritems() if u <= N],
                         key=itemgetter(1) )
    if len(availables) == 0: return []

    # hecks first if any jobs fits exactly the available number of nodes.
    if availables[-1][1] == N: return [availables[-1][0]]

    # creates a map of bins: 
    bins = {}
    for key, u in availables: 
      if u not in bins: bins[u] = 1
      else: bins[u] += 1

    def get(n, bins, xvec):
      """ Loops over possible combinations. """
      from random import choice
      key = choice(list(bins.keys()))
      for u in xrange(min(bins[key], n // key), -1, -1):
        newbins = bins.copy()
        del newbins[key]
        newn = n - u * key
        if newn == 0:
          yield xvec + [(key, u)], True
          break
        for v in list(newbins.keys()):
          if v > newn: del newbins[v]
        if len(newbins) == 0:
          yield xvec + [(key, u)], False
          continue
        for othervec, perfect in get(newn, newbins, xvec + [(key, u)]):
          yield othervec, perfect
          if perfect: return

    xvec = []
    nprocs, njobs = 0, 0
    for u, perfect in get(N, bins, []):
      if perfect: xvec = u; break
      p, j = sum(a*b for a, b in u), sum(a for a, b in u)
      if p > nprocs or (p == nprocs and j < njobs):
        xvec, nprocs, njobs = list(u), p, j

    # now we have a vector with the right number of jobs, but not what those
    # jobs are.
    results = []
    for key, value in xvec:
      withkeyprocs = [name for name, n in availables if n == key]
      results.extend(withkeyprocs[:value])
    return results
    
   
  def start(self, comm):
    """ Start executing job-folders. """
    from .process import Process
    from .mpi import MPISizeError
    if isinstance(self.processalloc, int): 
      self.nbpools = comm['n'] // self.processalloc
      return super(PoolProcess, self).start(comm)

    if Process.start(self, comm): return True

    # check max job size.
    toolarge = [key for key, u in self._alloc.iteritems() if u > comm['n']]
    if len(toolarge):
      raise MPISizeError( "The following jobs require too many processors:\n"\
                                "{0}\n".format(toolarge) )

    self._next()
    return False

  def update(self, jobfolder, deleteold=False):
    """ Updates list of jobs.
    
        Adds jobfolders which are not in ``self.jobfolder`` but in the input.
        Deletes those which in ``self.jobfolder`` but not in the input.
        Does nothing if job is currently running.
        Finished jobs are not updated.
        If ``deleteold`` is True, then removed finished jobs from job-folder.

        Processes jobfolder from root, even if passed a child folder.
    """
    running = set([n for n in self.process])
    for name, value in jobfolder.root.iteritems():
      if name in running: continue
      elif name not in self.jobfolder.root:
        newjob = self.jobfolder.root / name
        newjob.functional = value.functional
        newjob.params.update(value.params)
        for key, value in value.__dict__.iteritems():
          if key in ['children', 'params', '_functional', 'parent']: continue
          setattr(self, key, value)
        self._torun.add(name)
        self._alloc[name] = self.processalloc(self.jobfolder.root[name])
      elif name not in self._finished:
        self.jobfolder.root[name] = value
        self._alloc[name] = self.processalloc(self.jobfolder.root[name])
    for name in self.jobfolder.root.iterkeys():
      if name in self._finished and deleteold:
        del self.jobfolder.root[name]
        self._alloc.pop(name)
      elif name not in jobfolder.root:
        if name in running: continue
        del self.jobfolder.root[name]
        if name in self._torun: self._torun.remove(name)
        self._alloc.pop(name, None)



