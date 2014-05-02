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

""" Pylada plugin for IPython. """
__all__ = ['load_ipython_extension']

__pylada_is_loaded__ = False
""" Whether the Pylada plugin has already been loaded or not. """
def load_ipython_extension(ip):
  """Load the extension in IPython."""
  global __pylada_is_loaded__
  if not __pylada_is_loaded__:
    from types import ModuleType
    import pylada
    __pylada_is_loaded__ = True
    pylada.interactive = ModuleType('interactive')
    pylada.interactive.jobfolder = None
    pylada.interactive.jobfolder_path = None
    pylada.is_interactive = True

def unload_ipython_extension(ip):
  """ Unloads Pylada IPython extension. """
  ip.user_ns.pop('collect', None)
  ip.user_ns.pop('jobparams', None)
