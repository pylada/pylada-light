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
  from collections import namedtuple
  from pickle import loads, dumps
  from pylada.crystal.cppwrappers import Structure
  from pylada.vasp.incar._params import UParams
  from pylada.vasp.specie import U, nlep

  u = 0.25
  x, y = u, 0.25-u
  structure = Structure([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]) \
                       .add_atom(5.000000e-01, 5.000000e-01, 5.000000e-01, "A") \
                       .add_atom(5.000000e-01, 2.500000e-01, 2.500000e-01, "A") \
                       .add_atom(2.500000e-01, 5.000000e-01, 2.500000e-01, "A") \
                       .add_atom(2.500000e-01, 2.500000e-01, 5.000000e-01, "A") \
                       .add_atom(8.750000e-01, 8.750000e-01, 8.750000e-01, "B") \
                       .add_atom(1.250000e-01, 1.250000e-01, 1.250000e-01, "B") \
                       .add_atom(     x,     x,     x, "X") \
                       .add_atom(     x,     y,     y, "X") \
                       .add_atom(     y,     x,     y, "X") \
                       .add_atom(     y,     y,     x, "X") \
                       .add_atom(    -x,    -x,    -x, "X") \
                       .add_atom(    -x,    -y,    -y, "X") \
                       .add_atom(    -y,    -x,    -y, "X") \
                       .add_atom(    -y,    -y,    -x, "X") 
  Vasp = namedtuple('Vasp', ['species'])
  Specie = namedtuple('Specie', ['U'])
  vasp = Vasp({'A': Specie([U(2, 0, 0.5)]), 'B': Specie([U(2, 0, -0.5), nlep(2, 1, -1.0)]), 'X': Specie([])})
  a =\
"""\
LDAU = .TRUE.
LDAUPRINT = 0
LDAUTYPE = 2

LDUL1= 0 -1 0
LDUU1=   5.0000000000e-01   0.0000000000e+00  -5.0000000000e-01
LDUJ1=   0.0000000000e+00   0.0000000000e+00   0.0000000000e+00
LDUO1= 1 1 1

LDUL2= -1 -1 1
LDUU2=   0.0000000000e+00   0.0000000000e+00  -1.0000000000e+00
LDUJ2=   0.0000000000e+00   0.0000000000e+00   0.0000000000e+00
LDUO2= 1 1 2
"""
  assert a == UParams('off').incar_string(vasp=vasp, structure=structure)
  vasp = Vasp({'A': Specie([U(2, 0, 0.5)]), 'B': Specie([U(2, 0, -0.5), nlep(2, 2, -1.0, -3.0)]), 'X': Specie([])})
  a =\
"""\
LDAU = .TRUE.
LDAUPRINT = 1
LDAUTYPE = 2

LDUL1= 0 -1 0
LDUU1=   5.0000000000e-01   0.0000000000e+00  -5.0000000000e-01
LDUJ1=   0.0000000000e+00   0.0000000000e+00   0.0000000000e+00
LDUO1= 1 1 1

LDUL2= -1 -1 2
LDUU2=   0.0000000000e+00   0.0000000000e+00  -1.0000000000e+00
LDUJ2=   0.0000000000e+00   0.0000000000e+00  -3.0000000000e+00
LDUO2= 1 1 3
"""
  assert a == UParams('on').incar_string(vasp=vasp, structure=structure)
  vasp = Vasp({'A': Specie([]), 'B': Specie([]), 'X': Specie([])})
  assert '# no LDA+U/NLEP parameters' == UParams('all').incar_string(vasp=vasp, structure=structure)

  assert repr(UParams('off')) == "UParams('off')"
  assert repr(UParams('on')) == "UParams('on')"
  assert repr(UParams(None)) == "UParams('off')"
  assert repr(UParams('all')) == "UParams('all')"
  assert repr(loads(dumps(UParams('off')))) == "UParams('off')"
  assert repr(loads(dumps(UParams('on')))) == "UParams('on')"
  assert repr(loads(dumps(UParams(None)))) == "UParams('off')"
  assert repr(loads(dumps(UParams('all')))) == "UParams('all')"

if __name__ == "__main__":
  from sys import argv, path 
  from numpy import array
  if len(argv) > 0: path.extend(argv[1:])
  
  test()

