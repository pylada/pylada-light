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

""" Holds base classes and mixins for extraction objects. """
__docformat__ = "restructuredtext en"
__all__ = ['AbstractExtractBase', 'search_factory']
from abc import ABCMeta, abstractproperty

class AbstractExtractBase(object):
  """ Abstract base class for extraction classes. 
  
      Defines a number of members common to all extraction classes:
        - directory: root directory where output should exist.
  """
  __metaclass__ = ABCMeta
  def __init__(self, directory=None):
    """ Initializes an extraction base class.

        :Parameters: 
          directory : str or None
            Root directory for extraction. If None, will use current working directory.
    """
    from pylada.misc import bugLev

    if bugLev >= 5:
      print 'tools/extract: AbstractExtractBase: object: ', object
    object.__init__(self)

    from os import getcwd
    from ..misc import RelativePath

    if directory is None: directory = getcwd()
    self._directory = RelativePath(directory, hook=self.__directory_hook__)
    """ Directory where output should be found. """

  @property
  def directory(self):
    """ Directory where output should be found. """
    return self._directory.path
  @directory.setter
  def directory(self, value): self._directory.path = value

  @abstractproperty
  def success(self):
    """ Checks for success. 

        Should never ever throw!
        True if calculations were successfull, false otherwise.
    """
    pass


  def __directory_hook__(self):
    """ Called whenever the directory changes. """
    self.uncache()

  def uncache(self): 
    """ Uncache values. """
    self.__dict__.pop("_properties_cache", None)

  def __copy__(self):
    """ Returns a shallow copy of this object. """
    from ..misc import RelativePath
    result = self.__class__(directory=self.directory)
    result.__dict__ = self.__dict__.copy()
    result._directory = RelativePath( self._directory.path,\
                                           self._directory._envvar, 
                                           result.uncache )
    return result

  def copy(self, **kwargs):
    """ Returns a shallow copy of this object.

        :param kwargs:
          Any keyword argument is set as an attribute of this object.
          The attribute must exist.
    """
    from ..error import KeyError
    result = self.__copy__()
    for k, v in kwargs.iteritems():
      if not hasattr(self, k):
        raise KeyError('Attribute {0} does not exist.'.format(k))
      setattr(result, k, v)
    return result

  def __getstate__(self):
    d = self.__dict__.copy()
    if "_directory" in d: d["_directory"].hook = None
    return d

  def __setstate__(self, arg):
    self.__dict__.update(arg)
    if hasattr(self, "_directory"): self._directory.hook = self.uncache

  def __repr__(self):
    return "{0}(\"{1}\")".format(self.__class__.__name__, self._directory.unexpanded)

def search_factory(name, methname, module, filename=None):
  """ Factory to create Mixing classes capable of search a given file. """
  if filename is None: filename = methname.upper()
  doc = \
    """ A mixin to include standard methods to search {0}.
    
        This mixin only includes the search methods themselves. The derived
        class should define the appropriate {1} attribute. 
    """.format(filename, methname.upper())
  def __outcar__(self):
    """ Returns path to OUTCAR file.

        :raise IOError: if the OUTCAR file does not exist. 
    """
    from os.path import exists, join
    from ..error import IOError
    path = join(self.directory, getattr(self, methname.upper()))
    if not exists(path):
      raise IOError("Path {0} does not exist.\n".format(path))
    return open(path, 'r')
  __outcar__.__name__ = '__{0}__'.format(methname.lower())

  def _search_OUTCAR(self, regex, flags=0):
    """ Looks for all matches. """
    from re import compile, M as moultline

    regex  = compile(regex, flags)
    with getattr(self, __outcar__.__name__)() as file:
      if moultline & flags: 
        for found in regex.finditer(file.read()): yield found
      else:
        for line in file: 
          found = regex.search(line)
          if found is not None: yield found
  _search_OUTCAR.__name__ = '_search_{0}'.format(methname.upper())

  def _find_first_OUTCAR(self, regex, flags=0):
    """ Returns first result from a regex. """
    for first in getattr(self, _search_OUTCAR.__name__)(regex, flags): return first
    return None
  _find_first_OUTCAR.__name__ = '_find_first_{0}'.format(methname.upper())

  def _rsearch_OUTCAR(self, regex, flags=0):
    """ Looks for all matches starting from the end. """
    from re import compile, M as moultline

    regex  = compile(regex)
    with getattr(self, __outcar__.__name__)() as file:
      lines = file.read() if moultline & flags else file.readlines()
    if moultline & flags: 
      for v in [u for u in regex.finditer(lines)][::-1]: yield v
    else:
      for line in lines[::-1]:
        found = regex.search(line)
        if found is not None: yield found
  _rsearch_OUTCAR.__name__ = '_rsearch_{0}'.format(methname.upper())

  def _find_last_OUTCAR(self, regex, flags=0):
    """ Returns first result from a regex. """
    for last in getattr(self, _rsearch_OUTCAR.__name__)(regex, flags): return last
    return None
  _find_last_OUTCAR.__name__ = '_find_last_{0}'.format(methname.upper())

  attrs = { __outcar__.__name__: __outcar__,
            _search_OUTCAR.__name__: _search_OUTCAR,
            _rsearch_OUTCAR.__name__: _rsearch_OUTCAR,
            _find_first_OUTCAR.__name__: _find_first_OUTCAR,
            _find_last_OUTCAR.__name__: _find_last_OUTCAR,
            methname.upper(): filename,
            '__doc__': doc,
            '__module__': module }
  return type(name, (), attrs)
