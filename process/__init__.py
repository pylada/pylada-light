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

""" Manages external processes and jobfolders. 

    This sub-module provides an abstraction between resources which
    purports to launch external programs, say a
    :py:class:`~pylada.vasp.functional.Vasp` instance, and the actual
    program. There are two main issues the module attempts to resolve: 

      - an interface which hides the details of launching mpi jobs on one
        or another super-computer
      - an interface to launch different calculations in parallel but from
        a single actual system process, e.g. asynchronous management of
        different mpi-processes
    
    The first point above is easy to understand: some machines use openmpi_
    as is, others provide different flavors of mpich, and Cray use their
    own crap pseudo-proprietary software. It's best to keep all those
    details in one place. The second point is to make it easy to launch
    different calculations simultaneously. It provides an additional layer
    for parallelization, in addition to the one provided below at the
    application level by mpich and friends,  and above it by the queuing
    system of a particular super-computer. 

    The module is organized around :py:class:`~process.Process` and its
    derived classes. Instances of these classes are meant to be used as
    follows:

    .. code-block:: python

      process = SomeProcess()
      process.start(comm)
      # do something else
      try:
       if process.poll():
         # process did finish.
      except Fail as e: 
        # an error occured

    The first two lines initialize and start a process of some kind, which
    could be as simple as lauching an external :py:class:`program
    <program.ProgramProcess>` or as complicated as lauching different jobs
    from a :py:class:`~pylada.jobfolder.jobfolder.JobFolder` instance in
    :py:class:`parallel <jobfolder.JobFolderProcess>`. The rest of the
    snippet checks whether the process is finished. If it finished
    correctly, then :py:meth:`~process.Process.poll` (rather, the overloaded
    functions in the derived class) returns True. Otherwise, it throws
    :py:class:`Fail`.

    It is expected that processes will be returned (or yielded) by
    functionals, and then further managed by the script of interest. As such,
    the initialization of a process will depend upon the actual functional,
    and what it purports to do. However, it should be possible from then on
    to manage the process in a standard way. This standard interface is
    described by the abstract base class :py:class:`~process.Process`.

    .. _openmpi: http://www.open-mpi.org/
"""
__docformat__ = "restructuredtext en"
__all__  = [ 'Process', 'ProgramProcess', 'CallProcess', 'IteratorProcess',
             'JobFolderProcess', 'PoolProcess', 'Fail', 'which', 'DummyProcess' ]

from ..error import root
from pool import PoolProcess
from process import Process
from call import CallProcess
from program import ProgramProcess
from iterator import IteratorProcess
from jobfolder import JobFolderProcess
from dummy import DummyProcess

class ProcessError(root):
  """ Root of special exceptions issued by process module. """
class Fail(ProcessError):
  """ Process failed to run successfully. """
  pass
class AlreadyStarted(ProcessError):
  """ Process already started.
      
      Thrown when :py:meth:`~process.Process.start` or its overloaded friend is
      called for a second time.
  """
class NotStarted(ProcessError):
  """ Process was never started.
      
      Thrown when :py:meth:`~process.Process.poll` or its overloaded friend is
      called before the process is started.
  """

def which(program):
  """ Gets location of program by mimicking bash which command. """
  from os import environ, getcwd
  from os.path import split, expanduser, expandvars, join
  from itertools import chain
  from ..misc import RelativePath
  from ..error import IOError

  def is_exe(path):
    from os import access, X_OK
    from os.path import isfile
    return isfile(path) and access(path, X_OK)

  exprog = expanduser(expandvars(program))
  fpath, fname = split(exprog)
  if fpath:
    if is_exe(exprog): return RelativePath(exprog).path
  else:
    for dir in chain([getcwd()], environ["PATH"].split(':')):
      if is_exe(join(dir, exprog)): return RelativePath(join(dir, exprog)).path

  raise IOError('Could not find executable {0}.'.format(program))

