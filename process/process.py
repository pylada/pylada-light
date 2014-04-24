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

from abc import ABCMeta, abstractmethod

class Process(object):
  """ Abstract base class of all processes. 
  
      This class defines the interface for processes. Derived classes should
      overload :py:meth:`start`, :py:meth:`poll`, and :py:meth:`wait`. The
      first is called to actually launch the sub-process (for instance, an
      actual call to vasp in :py:class:`~pylada.process.program.ProgramProcess`).
      It receives a dictionary or :py:class:`Communicator` instance with a
      description of how the process should be launched, eg the number of
      processors, nodes, and so forth. At this point, an external child program
      will generally be running. The second function, :py:meth:`poll`, is
      called to check whether the sub-process, say VASP, is still running. It
      returns True if the process is finished. The last function is equivalent
      to calling :py:meth:`poll` until it returns True.

      Futhermore, a process can be :py:meth:`terminated <terminate>`,
      :py:meth:`killed <kill>` and :py:meth:`cleaned up <_cleanup>`.
      In general, a process is used as follows:

      .. code-block:: python
 
        # initialized
        process = SomeProcess()
        # started on a number of processors
        process.start(comm)
        # checked 
        try: 
          if process.poll():
            # finished, do something
        except Fail as e:
          # error, do something
  """ 
  __metaclass__ = ABCMeta
  def __init__(self, maxtrials=1, **kwargs):
    """ Initializes a process. """
    super(Process, self).__init__()

    self.nberrors = 0
    """ Number of times process was restarted.
        
        Some derived instances may well restart a failed sub-process. This is
        how often it has been restarted.
    """
    self.maxtrials = maxtrials
    """ Maximum number of restarts. """
    self.process = None
    """ Currently running process.
    
        This is the sub-process handled by this instance. At the lowest
        level, it is likely an instance of `subprocess.Popen`__. It may,
        however, be a further abstraction, such as a
        :py:class:`~pylada.process.program.ProgramProcess` instance.

        .. __ : http://docs.python.org/library/subprocess.html#subprocess.Popen
    """
    self.started = False
    """ Whether the process was ever started.
    
        Whether :py:meth:`start` was called. It may only be called once. 
    """

  @abstractmethod
  def poll(self): 
    """ Polls current job.
        
        :return: True if the process is finished.
        :raise NotStarted: If the process was never launched.
    """
    from . import NotStarted
    if not self.started: raise NotStarted("Process was never started.")
    return self.nbrunning_processes == 0

  @property 
  def done(self):
    """ True if job already finished. """
    return self.started and self.process is None

  @property
  def nbrunning_processes(self):
    """ Number of running processes. 

        For simple processes, this will be one or zero.
        For multitasking processes this may be something more.
    """
    return 0 if (not self.started) or self.process is None else 1

  @abstractmethod
  def start(self, comm):
    """ Starts current job. 
    
        :param comm: 
          Holds information about how to launch an mpi-aware process. 
        :type comm: :py:class:`~process.mpi.Communicator`

        :returns: True if process is already finished.
        
        :raises MPISizeError:
        
           if no communicator is not None and `comm['n'] == 0`. Assumes that
           running on 0 processors is an error in resource allocation.
        
        :raise AlreadyStarted:
        
           When called for a second time. Each process should be unique: we
           do not want to run the VASP program twice in the same location,
           especially not simultaneously.
    """
    from . import AlreadyStarted
    from .mpi import MPISizeError, Communicator
    if self.done: return True
    if self.started: raise AlreadyStarted('start cannot be called twice.')
    self.started = True
    if comm is not None:
      if comm['n'] == 0: raise MPISizeError('Empty communicator passed to process.')
      self._comm = comm if hasattr(comm, 'machines') else Communicator(**comm) 
    return False

  def _cleanup(self):
    """ Cleans up behind process instance.
    
        This may mean closing standard input/output file, removing temporary
        files.. By default, calls cleanup of :py:attr:`process`, and sets
        :py:attr:`process` to None.
    """
    try:
      if hasattr(self.process, '_cleanup'): self.process._cleanup()
    finally:
      self.process = None
      if hasattr(self, '_comm'): 
        try: self._comm.cleanup()
        finally: del self._comm

  def __del__(self):
    """ Kills process if it exists.
    
        Tries and cleans up this instance cleanly. 
        If a process exists, it is cleaned. The existence of this function
        implies that the reference to this instance should best not be lost
        until whatever it is supposed to oversee is finished doing its stuff.
    """
    if self.process is None: return
    try: self.process.kill()
    except: pass
    try: self._cleanup()
    except: pass


  def terminate(self):
    """ Terminates current process. """
    if self.process is None: return
    try: self.process.terminate()
    except: pass
    self._cleanup()

  def kill(self):
    """ Kills current process. """
    if self.process is None: return
    try: self.process.kill()
    except: pass
    self._cleanup()

  @abstractmethod
  def wait(self):
    """ Waits for process to end, then cleanup. """
    from . import NotStarted
    if not self.started: raise NotStarted("Process was never started.")
    if self.nbrunning_processes == 0: return True
