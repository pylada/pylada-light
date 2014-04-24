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

""" Holds exceptions declared by Pylada. """

class root(Exception):
  """ Root for all Pylada exceptions. """
  pass

class input(root):
  """ Root for all input Pylada exceptions. """
  pass

class out_of_range(root):
  """ Root for all out-of-range Pylada exceptions. """
  pass

class internal(root, RuntimeError):
  """ Root for all internal (cpp) Pylada exceptions. """
  pass

class infinite_loop(root):
  """ Root for all infinite-loops Pylada exceptions. """
  pass

class ValueError(root, ValueError):
  """ Root for all ValueError Pylada exceptions. """
  pass

class KeyError(root, KeyError):
  """ Root for all KeyError Pylada exceptions. """
  pass

class AttributeError(root, AttributeError):
  """ Root for all AttributeError Pylada exceptions. """
  pass

class IndexError(root, IndexError):
  """ Root for all IndexError Pylada exceptions. """
  pass

class TypeError(root, TypeError):
  """ Root for all TypeError Pylada exceptions. """
  pass

class NotImplementedError(root, NotImplementedError):
  """ Root for all NotImplementedError Pylada exceptions. """
  pass

class ImportError(root, ImportError):
  """ Root for all ImportError Pylada exceptions. """
  pass

class IOError(root, IOError):
  """ Root for all ImportError Pylada exceptions. """
  pass

class Math(root):
  """ Root of math exceptions. """
  pass
class singular_matrix(Math):
  """ Singular matrix. """
  pass
class interactive(input):
  """ Interactive usage error. """
  pass
class GrepError(AttributeError):
  """ Raised when property could not be grepped from some OUTCAR. """
  pass
class ConfigError(input):
  """ Some sort of Pylada configuration error. """

class ExternalRunFailed(root):
  """ Thrown when an external run has failed. """
