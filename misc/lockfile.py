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

""" I/O Lock for inter-process synchronization.

    Leverages the (presumed?) atomicity of mkdir on unices.
"""
__docformat__  = 'restructuredtext en'
__all__ = ['LockFile', 'open_exclusive']
from contextlib import contextmanager
  
class LockFile(object):
  """ Gets an advisory lock for file C{filename}.

      Creates a lock directory named after a file. Relies on the presumably
      atomic nature of creating a directory. 
      *Beware* of mpi problems! `LockFile` is (purposefully) not mpi aware.
      If used unwisely, processes will lock each other out.
  """
  def __init__(self, filename, timeout = None, sleep = 0.05):
    """ Creates a lock object. 

        :Parameters:
          timeout
            will raise a RuntimeError when calling `lock` if
            the lock could not be aquired within this time.
          sleep
            Time to sleep between checks when waiting to acquire lock. 

        Does not acquire lock at this stage. 
    """
    from os.path import abspath
    self.filename = abspath(filename)
    """ Name of file to lock. """
    self.timeout = timeout
    """ Maximum amount of time to wait when acquiring lock. """
    self.sleep = sleep
    """ Sleep time between checks on lock. """
    self._owns_lock = False
    """ True if this object owns the lock. """

  def lock(self):
    """ Waits until lock is acquired. """
    from os import makedirs, error, mkdir
    from os.path import exists
    import time

    # creates parent directory first, if necessary.
    if not exists(self._parent_directory):
      try: makedirs(self._parent_directory) 
      except error: pass
    start_time = time.time()
    # loops until acqires lock.
    while self._owns_lock == False: 
      # tries to create director.
      try:
        self._owns_lock = True
        mkdir(self.lock_directory)
      # if fails, then another process already created it. Just keep looping.
      except error: 
        self._owns_lock = False
        # 2013-11-11: disable timeout to make it try forever,
        # since timeouts are causing large runs to fail.
        #if self.timeout is not None:
        #  if time.time() - start_time > self.timeout:
        #    raise RuntimeError("Could not acquire lock on file {0}.".format(self.filename))
        time.sleep(self.sleep)

  def __del__(self):
    """ Deletes hold on object. """
    if self.owns_lock: self.release()
  def __enter__(self):
    """ Enters context. """
    self.lock()
    return self

  def __exit__(self, *args):
    """ Exits context. """
    self.release()

  @property
  def lock_directory(self):
    """ Name of lock directory. """
    from os.path import join, basename
    return join(self._parent_directory, "." + basename(self.filename) + "-pylada_lockdir")
 
  @property
  def _parent_directory(self):
    from os.path import abspath, dirname
    return dirname(abspath(self.filename))

  @property
  def is_locked(self):
    """ True if a lock for this file exists. """
    from os.path import exists
    return exists(self.lock_directory)

  @property
  def owns_lock(self): 
    """ True if this object owns the lock. """
    return self._owns_lock

  def release(self):
    """ Releases a lock.

        It is an error to release a lock not owned by this object.
        It is also an error to release a lock which is not locked.
        Makes sure things are syncro. The latter is an internal bug though.
    """
    from os import rmdir
    assert self._owns_lock, IOError("Filelock object does not own lock.")
    assert self.is_locked, IOError("Filelock object owns an unlocked lock.")
    self._owns_lock = False
    rmdir(self.lock_directory)

  def __del__(self):
    """ Releases lock if still held. """
    if self.owns_lock and self.is_locked: self.release()

  def remove_stale(self):
    """ Removes a stale lock. """
    from os import rmdir
    from os.path import exists
    if exists(self.lock_directory):
      try: rmdir(self.lock_directory)
      except: pass

def acquire_lock(filename, sleep=0.5, timeout=None):
  """ Alias for a `LockFile` context. 

      *Beware* of mpi problems! `LockFile` is (purposefully) not mpi aware.
      Only the root node should use this method.
  """
  return LockFile(filename, sleep=sleep, timeout=timeout)

@contextmanager
def open_exclusive(filename, mode="r", sleep = 0.5, timeout=None):
  """ Opens file while checking for advisory lock.

      This context uses `LockFile` to first obtain a lock.
      *Beware* of mpi problems! `LockFile` is (purposefully) not mpi aware.
      Only the root node should use this method.
  """
  # Enter context.
  with LockFile(filename, sleep=sleep, timeout=timeout) as lock:
    yield open(filename, mode)

