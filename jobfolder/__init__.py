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

""" Classes and functions pertaining to job-management. 

    The  :py:mod:`pylada.jobfolder` provides tools for high-throughput calculations.
    It is centered around an object - the job-folder - which organizes calculations
    within a tree of folders, much as one would manually organize calculations
    within a tree of directories. Each folder can be executable, e.g.  there is
    something to compute there, or non-executable. And each folder can further hold
    any number of sub-folders. Furthermore, classes are provided which make it easy
    to manipulate the parameters for the calculations in an executable folder, as
    well as within all subfolders. Finally, a similar infrastructure is provided to
    collect the computational results across all executable sub-folders.

    .. seealso:: :ref:`jobfolder_ug`
"""
__docformat__ = "restructuredtext en"
__all__ = ['JobFolder', 'walk_through', 'save', 'load', 'MassExtract',
           'AbstractMassExtract', 'JobParams' ]

from .jobfolder import JobFolder
from .manipulator import JobParams
from .extract import AbstractMassExtract
from .massextract import MassExtract 

def save(jobfolder, path='jobfolder.dict', overwrite=False, timeout=None): 
  """ Pickles a job-folder to file. 
 
      :param jobfolder:
          A job-dictionary to pickle. 
      :type jobfolder: :py:class:`~jobfolder.JobFolder` 
      :param str path: 
          filename of file to which to save pickle. overwritten. If None then
          saves to "pickled_jobfolder"
      :param int timeout: 
         How long to wait when trying to acquire lock on file.
         Defaults to forever.
      :param bool overwrite:
          if True, then overwrites file.

      This method first acquire an exclusive lock on the file before writing
      (see :py:meth:`pylada.misc.open_exclusive`).  This way not two processes can
      read/write to this file while using this function.
  """ 
  from os.path import exists
  from pickle import dump
  from ..misc import open_exclusive, RelativePath
  from .. import is_interactive
  path = RelativePath(path).path
  if exists(path) and not overwrite: 
    if is_interactive:
      print path, "exists. Please delete first if you want to save the job folder."
      return
    else: raise IOError('{0} already exists. By default, will not overwrite.'.format(path))
  with open_exclusive(path, "wb", timeout=timeout) as file: dump(jobfolder, file)
  if is_interactive: print "Saved job folder to {0}.".format(path)

def load(path='jobfolder.dict', timeout=None): 
  """ Unpickles a job-folder from file. 
 
      :param str path: 
         Filename of a pickled job-folder. 
      :param int timeout: 
         How long to wait when trying to acquire lock on file.
         Defaults to forever.
      :return: Returns a JobFolder object.

      This method first acquire an exclusive lock on the file before reading.
      This way not two processes can read/write to this file while using this
      function.
  """ 
  from os.path import exists
  from pickle import load as load_pickle
  from ..misc import open_exclusive, RelativePath
  from .. import is_interactive
  path = "job.dict" if path is None else RelativePath(path).path
  if not exists(path): raise IOError("File " + path + " does not exist.")
  with open_exclusive(path, "rb", timeout=timeout) as file: result = load_pickle(file)
  if is_interactive: print "Loaded job list from {0}.".format(path)
  return result
