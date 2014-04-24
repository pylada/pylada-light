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

__docformat__ = "restructuredtext en"
__all__ = ['Tree']

class Tree(list):
  """ Tree structure. 
  
      A tree structure is an enhanced list which can refer to its items both in
      terms of position and in terms of a keyword, as in a dictionary.
      Furthermore, it can branch into further Tree instances. The branches can
      be reached/created easily via :py:method:`descend`.

      It is possible to create several items with the same name using the
      method ``append``. However, this can cause some confusion since 
      :py:method:`__getitem__`, :py:method:`__setitem__`, :py:method:`descend`
      will act upon the first item with that particular name when indexed with
      a string.

      .. warning:: not exactly a fast container. 
  """
  __slots__ = ['prefix', 'suffix']
  def __init__(self, *args, **kwargs):
    super(Tree, self).__init__()
    if 'prefix' in kwargs:  self.prefix = kwargs.pop('prefix')
    if 'suffix' in kwargs:  self.suffix = kwargs.pop('suffix')
    for key, value in args: self.append((key, value))
    for key, value in kwargs.iteritems(): self.append((key, value))
  def descend(self, *args):
    """ Creation/Access to Tree instances within the list. 
    
        Each argument descends one level in the tree. If the branch does not
        exist, it is created.
    """
    from ...error import IndexError
    if len(args) == 0: return self
    name, args = args[0], args[1:]
    for key, value in self: 
      if name == key:
        if hasattr(value, 'descend'):
           return value.descend(*args)
        elif len(args) == 0: return value
        else:
          raise IndexError( 'The key {0} exists but is not a Tree instance.\n' \
                            'It is not possible to descend to {1[0]}'          \
                            .format(key, args) )
    self.append((name, Tree()))
    return self[-1][1].descend(*args)
  def __getitem__(self, name):
    from ...error import KeyError
    if isinstance(name, str): 
      for key, value in self: 
        if name == key: return value
      raise KeyError('Unknown key {0}'.format(name))
    return super(Tree, self).__getitem__(name)
  def __setitem__(self, name, value):
    from ...error import ValueError
    if isinstance(name, str):
      for i, (key, v) in enumerate(self):
        if name == key: 
          super(Tree, self).__setitem__(i, (name, value))
          return
      self.append((name, value))
      return 
    if not hasattr(value, '__len__'):
      raise ValueError('Expected a 2-tuple consisting of key-value pair.')
    if len(value) != 2:
      raise ValueError('Expected a 2-tuple consisting of key-value pair.')
    if not isinstance(value[0], str):
      raise ValueError( 'Expected a 2-tuple consisting of key-value pair, '    \
                        'with the key a string instance.' )
    super(Tree, self).__setitem__(name, value)
  def keys(self): return [u[0] for u in self]
  def iterkeys(self): 
    for key, value in self: yield key
  def values(self): return [u[1] for u in self]
  def itervalues(self): 
    for key, value in self: yield value
  def items(self): return self 
  def iteritems(self): return self.__iter__()
  def update(self, value):
    if value is None: return
    if isinstance(value, dict): value = value.iteritems()
    for key, value in value: self.append((key, value))
  def __contains__(self, index):
    if isinstance(index, str):
      for key, value in self:
        if key == index: return True
      return False
    return super(Tree, self).__contains__(index)
