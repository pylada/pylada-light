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

""" Mixin classes for extraction objects. """
__docformat__  = 'restructuredtext en'
from ...tools.extract import search_factory
OutcarSearchMixin = search_factory('OutcarSearchMixin', 'OUTCAR', __name__)


class IOMixin(OutcarSearchMixin):
  """ A mixin base clase which controls file IO. 

      Defines special property with file-like behaviors. 
      Makes it easier to change the behavior of the extraction class.
  """
  def __init__(self, directory=None, OUTCAR=None, FUNCCAR=None, CONTCAR=None):
    """ Initializes the extraction class. 

        :Parameters: 
          directory : str or None
            path to the directory where the VASP output is located. If none,
            will use current working directory. Can also be the path to the
            OUTCAR file itself. 
          OUTCAR : str or None
            If given, this name will be used, rather than files.OUTCAR.
          CONTCAR : str or None
            If given, this name will be used, rather than files.CONTCAR.
    """
    from .. import files
    
    object.__init__(self)

    self.OUTCAR  = OUTCAR if OUTCAR is not None else files.OUTCAR
    """ Filename of the OUTCAR file from VASP. """
    self.CONTCAR  = CONTCAR if CONTCAR is not None else files.CONTCAR
    """ Filename of the CONTCAR file from VASP. """
    OutcarSearchMixin.__init__(self)

  def __contcar__(self):
    """ Returns path to FUNCCAR file.

        :raise IOError: if the FUNCCAR file does not exist. 
    """
    from os.path import exists, join
    path = join(self.directory, self.CONTCAR)
    if not exists(path): raise IOError("Path {0} does not exist.\n".format(path))
    return open(path, 'r')
 
  @property
  def is_running(self):
    """ True if program is running on this functional. 
         
        A file '.pylada_is_running' is created in the output folder when it is
        set-up to run CRYSTAL_. The same file is removed when CRYSTAL_ returns
        (more specifically, when the :py:class:`pylada.process.ProgramProcess` is
        polled). Hence, this file serves as a marker of those jobs which are
        currently running.
    """
    from os.path import join, exists
    from pylada.misc import bugLev
    is_run = exists(join(self.directory, '.pylada_is_running'))
    if bugLev >= 5: print 'vasp/extract/mixin: is_run: dir: %s  is_run: %s' \
      % (self.directory, is_run,)
    return is_run
