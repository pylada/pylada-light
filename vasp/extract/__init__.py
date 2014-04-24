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

""" Subpackage containing extraction methods for vasp parameters from vasp output. 

    Extaction objects are implemented as a mix and mash of bases classes. The
    reason for this is we want to isolate functionality specific to DFT and GW,
    and specific to reading *real* OUTCAR files and *database* OUTCAR files. 
"""
__docformat__  = 'restructuredtext en'
__all__ = ['Extract', 'MassExtract']
from ...tools.extract import AbstractExtractBase
from .base import ExtractBase
from .mixin import IOMixin
from ...jobfolder import AbstractMassExtract

class Extract(AbstractExtractBase, IOMixin, ExtractBase):
  """ Extracts DFT data from an OUTCAR. """
  def __init__(self, directory=None, **kwargs):
    """ Initializes extraction object. 
    
        :param directory: 
          Directory where the OUTCAR resides. 
          It may also be the path to an OUTCAR itself, if the file is not
          actually called OUTCAR.
    """
    from os.path import exists, isdir, basename, dirname
    from ...misc import RelativePath
       
    outcar = None
    if directory is not None:
      directory = RelativePath(directory).path
      if exists(directory) and not isdir(directory):
        outcar = basename(directory)
        directory = dirname(directory)
    AbstractExtractBase.__init__(self, directory)
    ExtractBase.__init__(self)
    IOMixin.__init__(self, directory, OUTCAR=outcar, **kwargs)
  @property
  def success(self):
    """ True if calculation was successfull. """
    return ExtractBase.success.__get__(self)


class MassExtract(AbstractMassExtract):
  """ Extracts all Vasp calculations in directory and sub-directories. 
    
      Trolls through all subdirectories for vasp calculations, and organises
      results as a dictionary where keys are the name of the diretories.

      Usage is simply:

      >>> from pylada.vasp import MassExtract
      >>> a = MassExtract('path') # or nothing if path is current directory.
      >>> a.success
      {
        '/some/path/':      True,
        '/some/other/path': True
      }
      >>> a.eigenvalues
  """
  def __init__(self, path = None, **kwargs):
    """ Initializes MassExtract.
    
    
        :Parameters:
          path : str or None
            Root directory for which to investigate all subdirectories.
            If None, uses current working directory.
          kwargs : dict
            Keyword parameters passed on to AbstractMassExtract.

        :kwarg naked_end: True if should return value rather than dict when only one item.
        :kwarg unix_re: converts regex patterns from unix-like expression.
    """
    from os import getcwd
    if path is None: path = getcwd()
    # this will throw on unknown kwargs arguments.
    super(MassExtract, self).__init__(path=path, **kwargs)

  def __iter_alljobs__(self):
    """ Goes through all directories with an OUTVAR. """
    from os import walk
    from os.path import relpath, join
    from . import Extract as VaspExtract
    from ..relax import RelaxExtract

    for dirpath, dirnames, filenames in walk(self.rootpath, topdown=True, followlinks=True):
      if 'OUTCAR' not in filenames: continue
      if 'relax_cellshape' in dirnames or 'relax_ions' in dirnames:
        dirnames[:] = [u for u in dirnames if u not in ['relax_cellshape', 'relax_ions']]
        try: result = RelaxExtract(join(self.rootpath, dirpath))
        except:
          try: result = VaspExtract(join(self.rootpath, dirpath))
          except: continue
      else: 
        try: result = VaspExtract(join(self.rootpath, dirpath))
        except: continue

      yield join('/', relpath(dirpath, self.rootpath)), result

  def __copy__(self):
    """ Returns a shallow copy. """
    result = self.__class__(self.rootpath)
    result.__dict__.update(self.__dict__)
    return result

  @property
  def _attributes(self): 
    """ Returns __dir__ set special to the extraction itself. """
    from . import Extract as VaspExtract
    return list(set([u for u in dir(VaspExtract) if u[0] != '_'] + ['details']))
  
