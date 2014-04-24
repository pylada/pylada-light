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

def test():
  from pylada.tools.input import Tree
  from pylada.error import ValueError

  a = Tree()
  assert len(a) == 0
  a = Tree(key=5)
  assert len(a) == 1
  assert a[0] == ('key', 5)
  assert a['key'] == 5
  for key in a.iterkeys(): assert key == 'key'
  for value in a.itervalues(): assert value == 5
  for key, value in a.iteritems(): assert key == 'key' and value == 5

  a = Tree(('key', 5), ('other', 'a'))
  assert len(a) == 2
  assert a[0] == ('key', 5)
  assert a['key'] == 5
  assert a[1] == ('other', 'a')
  assert a['other'] == 'a'
  iterator = a.iterkeys()
  assert iterator.next() == 'key'
  assert iterator.next() == 'other'
  try: iterator.next()
  except StopIteration: pass
  else: raise Exception()

  v = a.descend('branch', 'leaf')
  assert isinstance(v, Tree)
  assert isinstance(a['branch'], Tree)
  assert isinstance(a['branch'], Tree)
  assert isinstance(a['branch']['leaf'], Tree)
  assert a['branch']['leaf'] is v
  assert a[2][0] == 'branch'
  assert a[2][1] is a['branch']
  a['key'] = 6
  assert a['key'] == 6

  try: a[0] = 5
  except ValueError: pass
  else: raise Exception()
  try: a[0] = 5, 6
  except ValueError: pass
  else: raise Exception()


if __name__ == '__main__':
  test()
