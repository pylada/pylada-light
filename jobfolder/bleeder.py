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

""" Functor to *bleed* a job folder amongst pools of processes.
    
    Bleeding a job folder means that pools of processes are actively
    picking jobs from the dictionary, marking them as executed, and running
    them. Any one job should only be visited once.
"""
__docformat__ = "restructuredtext en"
__all__ = ['findone', 'iter', 'execall']
from abc import ABCMeta, abstractmethod


@contextmanager
def find_one(path, jobname=None):
  """ Returns an executabe job-folder as a context. 

      The goal is to retrieve an executable job-folder from disk, while using a
      lock-file to prevent other processes from reading/writing the job-folder
      at an importune time. The lock-file is not held throughout the existence
      of the context. It is only held for specific operations when entering and
      leaving the context. The function is a context_. As such, it is possible
      to modify the job-folder. The modifications will be saved when leaving
      the context.

      .. _context: http://docs.python.org/reference/datamodel.html#context-managers
  """ 
  from ..misc import LockFile

  found = False
  # acquire lock. Yield context outside of lock!
  with LockFile(path) as lock:
    # Loads pickle.
    with open(self._filename, 'r') as file: jobfolder = pickle_load(file)
    # Finds first untagged job.
    if jobname is None:
      for job in jobfolder.itervalues():
        if not job.is_tagged:
          found = True
          break
    else: job = jobfolder[jobname]
    # tag the job before surrendering the lock.
    if found and jobname is None:
      job.tag()
      with open(self._filename, 'w') as file: dump(jobfolder, file)

  # Check we found an untagged job. Otherwise, we are done.
  if not found:
    yield None
    return
    
  # context returns jobs.
  yield job

  # saves job since it might have been modified
  # acquire a lock first.
  with LockFile(self._filename) as lock:
    # Loads pickle.
    with open(self._filename, 'r') as file: jobfolder = pickle_load(file)
    # modifies job.
    jobfolder[job.name] = job
    # save jobs.
    with open(self._filename, 'w') as file: dump(jobfolder, file)

def iterall(path):
  """ Iterates over executable job-folders on disk. 
  
      The job-folder is continuously read from disk (and locked when
      read/written). This way, many processes can share access to the same
      job-folder.
  """
  from tempfile import NamedTemporaryFile
  from pickle import dump
  from ..opt import RelativeDirectory

  from os.path import exists
  from pickle import load as pickle_load, dump
  from ..misc import LockFile
  from ..error import IOError

  # tries to read file.
  if not exists(path): raise IOError('Job-folder does not exist.')
  # infinite loop. breaks when no new jobs can be found.
  while True:
    # only local root reads stuff. 
    job = None
    with findone(path) as job:
      if job is None: break
      yield job

class ExecAll(object):
  """ Executes job-folders on disk. """
  def __init__(self, comm, nbpools=1, retrial=2, folders=None):
    super(ExecAll, self).__init__()
    self.nbpools = nbpools
    """ Number of parallel processes. """
    self.folders = set([])
    """ Path to executable jobfolders. """
    if isintance(folders, list): self.folders = list(folders)
    elif folders is not None: self.folders = [folders]
    self.processes = []
    """ List of currently running processes. """
    self.finished = []
    """ Finished job-folders. """
    self.comm = comm 
    """ Communicator for each job. """
  def add_folder(self, path):
    """ Adds job-folder path to list. """
    from misc import RelativePath
    self.folders.add(RelativePath(path).path)

  def next(self, exhaust=False):
    """ Executes a job-folder if one is still available. """
    from random import choice
    # loop over current jobs.
    for i, (folderpath, jobname, trial, iterator, process) in enumerate(list(self.processes)): 
      poll = process.poll()
      if poll is None: continue
      self.pop(i)
      if poll < 0 and trial < self.trial: # ran into error, retrial.
        self.launch_process(folderpath, jobname, trial+1, iterator)
      else: self.launch_process(folderpath, jobname, trial+1, iterator)

    # add as many jobs as possible.
    if len(folders) == 0: return True
    while len(self.processes) < self.nbpools:
      folderpath = choice(self.folders)
      with findone(folderpath) as job:
        if job != None: self.launch_process(folderpath, job.name, trial, None)
        else: 
          i = self.folders.index(folderpath)
          self.finished.append(self.folderpath.pop(i))
    # If exhaust is True, then returns True if no more processes are running
    # and no more jobs left to execute.
    # If exhaust is False, then returns True if no more jobs left to execute
    # and fewer jobs than pools are left.
    return len(self.processes) == 0 if exhaust else len(self.processes) != self.nbpools
         
  def launch_process(self, folderpath, jobname, trial, iterator):
    """ Adds a process to the list. """
    from ..misc import Program
    if iterator is None:
      with findone(folderpath, jobname) as job:
        if hasattr(job.functional, 'iter'): 
          iterator = job.functional(comm=self.comm, **job.params)
          found = False
          for program in iterator:
            if not getattr(program, 'success', False): 
              found = True
              break
          if not found: return
        else: program = Program(







def execall(path, nbpools=1, processes=None, waittime=2, **kwargs):
  """ Executes job-folders on disk. """
  # loop over executable folders, without risk of other processes detrimentally
  # accessing the file on disk.
  if processes is None: processes = []
  for job in iterall(path):
    if len(processes) >= nbpools:


    
