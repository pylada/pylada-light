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
  from pylada.tools.makeclass import makeclass
  import dummy

  base = dummy.Base(1, 'b')

  Functional = makeclass('Functional', dummy.Base, dummy.iter_func, call=dummy.func)
  functional = Functional(b=-5, other=False)
  assert getattr(functional, 'other', True) == False
  assert functional(True) == False
  assert functional.a == 2
  assert dummy.iterator == 4
  assert functional.b == -5

  functional = Functional(b=-5, other=False, copy=base)
  assert functional.a == 1
  assert getattr(functional, 'other', True) == False
  assert functional(True) == False
  assert functional.a == 1
  assert dummy.iterator == 4
  assert functional.b == -5

  Functional = makeclass('Functional', dummy.Base, dummy.iter_func)
  functional = Functional(b=-5, other=False)
  assert getattr(functional, 'other', True) == False
  assert functional(True) is None
  assert functional.a == 2
  assert dummy.iterator == 4
  assert functional.b == -5

  functional = Functional(b=-5, other=False, copy=base)
  assert functional.a == 1
  assert getattr(functional, 'other', True) == False
  assert functional(True) is None
  assert functional.a == 1
  assert dummy.iterator == 4
  assert functional.b == -5

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  
  test()
