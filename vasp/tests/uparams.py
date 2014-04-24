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
  from numpy import all, abs, array
  from pylada.crystal import Structure
  from pylada.vasp import Vasp
  from pylada.vasp.specie import U, nlep
  import pylada

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
  a = Vasp()
  Specie = namedtuple('Specie', ['U'])
  a.species = {'A': Specie([]), 'B': Specie([]), 'X': Specie([])}
  pylada.vasp_has_nlep  = False

  o = a._input['ldau']
  d = {'LDAU': o.__class__}
  assert a.ldau == True
  assert o.output_map(vasp=a, structure=structure) is None
  assert eval(repr(o), d)._value == True
  assert eval(repr(o), d).keyword == 'LDAU'
  assert loads(dumps(o)).keyword == 'LDAU'
  assert loads(dumps(o))._value 

  # now disables U.
  a.species = {'A': Specie([U(2, 0, 0.5)]), 'B': Specie([]), 'X': Specie([])}
  a.ldau = False
  assert a.ldau == False
  assert o.output_map() is None
  # now prints U
  a.ldau = True
  map = o.output_map(vasp=a, structure=structure)
  assert map['LDAU'] == '.TRUE.'
  assert map['LDAUTYPE'] == '2'
  assert all(abs(array(map['LDUJ'].split(), dtype='float64')) < 1e-8)
  assert all(abs(array(map['LDUU'].split(), dtype='float64')-[0.5, 0, 0]) < 1e-8)
  assert all(abs(array(map['LDUL'].split(), dtype='float64')-[0, -1, -1]) < 1e-8)
  a.species = {'A': Specie([U(2, 0, 0.5)]), 'B': Specie([U(2, 1, 0.6)]), 'X': Specie([])}
  map = o.output_map(vasp=a, structure=structure)
  assert map['LDAU'] == '.TRUE.'
  assert map['LDAUTYPE'] == '2'
  assert all(abs(array(map['LDUJ'].split(), dtype='float64')) < 1e-8)
  assert all(abs(array(map['LDUU'].split(), dtype='float64')-[0.5, 0, 0.6]) < 1e-8)
  assert all(abs(array(map['LDUL'].split(), dtype='float64')-[0, -1, 1]) < 1e-8)
  

  # now tries NLEP
  pylada.vasp_has_nlep = True
  a.species = {'A': Specie([U(2, 0, 0.5)]), 'B': Specie([U(2, 0, -0.5), nlep(2, 1, -1.0)]), 'X': Specie([])}
  a.ldau = False
  assert a.ldau == False
  assert o.output_map() is None
  a.ldau = True
  map = o.output_map(vasp=a, structure=structure)
  assert map['LDAU'] == '.TRUE.'
  assert map['LDAUTYPE'] == '2'
  assert all(abs(array(map['LDUL1'].split(), dtype='float64')-[0, -1, 0]) < 1e-8)
  assert all(abs(array(map['LDUU1'].split(), dtype='float64')-[0.5, 0, -0.5]) < 1e-8)
  assert all(abs(array(map['LDUJ1'].split(), dtype='float64')-[0, 0, 0]) < 1e-8)
  assert all(abs(array(map['LDUO1'].split(), dtype='float64')-[1, 1, 1]) < 1e-8)
  assert all(abs(array(map['LDUL2'].split(), dtype='float64')-[-1, -1, 1]) < 1e-8)
  assert all(abs(array(map['LDUU2'].split(), dtype='float64')-[0, 0, -1.0]) < 1e-8)
  assert all(abs(array(map['LDUJ2'].split(), dtype='float64')-[0, 0, 0]) < 1e-8)
  assert all(abs(array(map['LDUO2'].split(), dtype='float64')-[1, 1, 2]) < 1e-8)

  a.species = {'A': Specie([U(2, 0, 0.5)]), 'B': Specie([U(2, 0, -0.5), nlep(2, 2, -1.0, -3.0)]), 'X': Specie([])}
  a.ldau = True
  map = o.output_map(vasp=a, structure=structure)
  assert map['LDAU'] == '.TRUE.'
  assert map['LDAUTYPE'] == '2'
  assert all(abs(array(map['LDUL1'].split(), dtype='float64')-[0, -1, 0]) < 1e-8)
  assert all(abs(array(map['LDUU1'].split(), dtype='float64')-[0.5, 0, -0.5]) < 1e-8)
  assert all(abs(array(map['LDUJ1'].split(), dtype='float64')-[0, 0, 0]) < 1e-8)
  assert all(abs(array(map['LDUO1'].split(), dtype='float64')-[1, 1, 1]) < 1e-8)
  assert all(abs(array(map['LDUL2'].split(), dtype='float64')-[-1, -1, 2]) < 1e-8)
  assert all(abs(array(map['LDUU2'].split(), dtype='float64')-[0, 0, -1.0]) < 1e-8)
  assert all(abs(array(map['LDUJ2'].split(), dtype='float64')-[0, 0, -3.0]) < 1e-8)
  assert all(abs(array(map['LDUO2'].split(), dtype='float64')-[1, 1, 3]) < 1e-8)


if __name__ == "__main__":
  test()

