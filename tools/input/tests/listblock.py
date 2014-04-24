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
  from pickle import loads, dumps
  from pylada.tools.input import ListBlock, BaseKeyword

  a = ListBlock()
  d = {'ListBlock': a.__class__}
  assert len(a) == 0
  assert isinstance(eval(repr(a), d), ListBlock)
  assert len(eval(repr(a), d)) == 0
  assert isinstance(loads(dumps(a)), ListBlock)
  assert len(loads(dumps(a))) == 0
  a.append('hello', True)
  assert len(a) == 1
  assert a[0].__class__ is BaseKeyword
  assert a[0].keyword == 'hello'
  assert a[0].raw == True
  assert a.output_map() == [('hello', True)]
  assert isinstance(eval(repr(a), d), ListBlock)
  assert len(eval(repr(a), d)) == 1
  assert eval(repr(a), d)[0].__class__ is BaseKeyword
  assert eval(repr(a), d)[0].keyword == 'hello'
  assert eval(repr(a), d)[0].raw == True
  assert len(loads(dumps(a))) == 1
  assert loads(dumps(a))[0].__class__ is BaseKeyword
  assert loads(dumps(a))[0].keyword == 'hello'
  assert loads(dumps(a))[0].raw == True
  b = ListBlock()
  b.read_input(a.output_map())
  assert repr(b) == repr(a)
  

if __name__ == '__main__':
  test()
