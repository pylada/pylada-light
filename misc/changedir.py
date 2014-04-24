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

""" Class to change working directory within local context only.
   
    >>> with Changedir(path) as pwd:
    >>>   ...
"""
class Changedir:
  """ Works with "with" statement to temporarily change the working directory 
   
      >>> with Changedir(path) as pwd:
      >>>   ...
  """
  def __init__(self, pwd): 
    self.pwd = pwd

  def __enter__(self):
    """ Changes working directory """
    from os import getcwd, chdir, makedirs
    from os.path import exists, isdir
    
    self.oldpwd = getcwd()

    if not exists(self.pwd): makedirs(self.pwd)
    if not exists(self.pwd):
      raise IOError("Could not create working directory {0}".format(self.pwd))
    if not isdir(self.pwd):
      raise IOError("{0} is not a directory.".format(self.pwd))
    chdir(self.pwd)

    return self.pwd

  def __exit__(self, type, value, traceback):
    """ Moves back to old pwd """
    from os import chdir
    from os.path import exists, isdir
    if not (exists(self.oldpwd) or not isdir(self.oldpwd)):
      raise IOError("Old directory does not exist anymore.")
    chdir(self.oldpwd)
