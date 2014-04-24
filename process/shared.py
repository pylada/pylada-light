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
class SharedJobFolderProcess(IteratorProcess):
  """ Executes jobfolder with more than one job in child process. """
  def __init__(self, jobfolderpath, maxtrials=1, comm=None, nbpools=1, **kwargs):
    """ Initializes a process. """
    from copy import deepcopy
    super(JobFolderProcess, self).__init__(maxtrials, comm, **kwargs)
    self.nbpools = nbpools
    """ Number of pools over which to separate communicator. """
    self.jobfolderpath = jobfolderpath
    # start first process.
    self.poll()

  def poll(): 
    """ Polls current job. """
    from subprocess import Popen
    from shlex import split as split_cmd
    from misc import Changedir
    from . import Program
    from pylada.misc import testValidProgram

    # check if we have currently running process.
    # if current process is finished running, closes stdout and stdout.
    if self.current_program[0] is not None: 
      if self.current_program[0].poll() is None: return
      # kills stdout and stderr.
      if hasattr(self.current_program[1].stdout, 'close'): 
        self.current_program[1].stdout.close()
      if hasattr(self.current_program[1].stderr, 'close'): 
        self.current_program[1].stderr.close()
      # now remove reference to current program.
      self.current_program = None, None

    # At this point, loop until find something to do.
    found = False
    params = self.jobfolder.params.copy()
    params.update(self.params)
    for i, program in self.jobfolder.iterator(**params):
      if not getattr(program, 'success', False): 
        found = True
        break;
    # stop if no more jobs.
    if found == False: raise IteratorProcess.StopIteration()
    # if stopped on or before previous job, then we have a retrial.
    if i <= self.iter_index:
      if self.nberrors >= self.maxtrials: raise IteratorProcess.StopIteration()
      self.nberrors += 1
    # Open stdout and stderr if necessary.
    with Changedir(program.directory) as cwd:
     file_out = open(program.stdout, "a" if append else "w") \
                if program.stdout is not None else None 
     file_err = open(program.stderr, "a" if append else "w") \
                if program.stdout is not None else None 

    # now create process.
    program = Program(program.program, program.cmdline, program.directory, fileout, filerr)
    process = Popen(split_cmd(cmd), stdout=file_out, stderr=file_err, cwd=program.directory)
    if testValidProgram: process.wait()
    self.current_program = process, program
