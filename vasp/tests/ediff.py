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

def test_ediff():
  from pickle import loads, dumps
  from pylada.vasp import Vasp
  from pylada.crystal import Structure
  a = Vasp()

  u = 0.25
  x, y = u, 0.25-u
  structure = Structure([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]) \
                       .add_atom(5.000000e-01, 5.000000e-01, 5.000000e-01, "Mg") \
                       .add_atom(5.000000e-01, 2.500000e-01, 2.500000e-01, "Mg") \
                       .add_atom(2.500000e-01, 5.000000e-01, 2.500000e-01, "Mg") \
                       .add_atom(2.500000e-01, 2.500000e-01, 5.000000e-01, "Mg") \
                       .add_atom(8.750000e-01, 8.750000e-01, 8.750000e-01, "Al") \
                       .add_atom(1.250000e-01, 1.250000e-01, 1.250000e-01, "Al") \
                       .add_atom(     x,     x,     x, "O") \
                       .add_atom(     x,     y,     y, "O") \
                       .add_atom(     y,     x,     y, "O") \
                       .add_atom(     y,     y,     x, "O") \
                       .add_atom(    -x,    -x,    -x, "O") \
                       .add_atom(    -x,    -y,    -y, "O") \
                       .add_atom(    -y,    -x,    -y, "O") \
                       .add_atom(    -y,    -y,    -x, "O") 
  
  N = float(len(structure))

  o = a._input['ediff']
  d = {'Ediff': o.__class__}
  assert a.ediff is None
  assert a.ediff_per_atom is None
  assert o.output_map() is None
  assert eval(repr(o), d).output_map() is None
  assert eval(repr(o), d).keyword == 'ediff'
  assert loads(dumps(o)).output_map() is None
  a.ediff_per_atom = 1e-5
  a.ediff = 2e-4
  assert abs(a.ediff - 2e-4) < 1e-8
  assert a.ediff_per_atom is None
  assert abs(float(o.output_map(structure=structure)['ediff']) - a.ediff) < a.ediff * 1e-2
  assert abs(float(eval(repr(o), d).output_map(structure=structure)['ediff']) - a.ediff) < a.ediff * 1e-2
  assert abs(float(loads(dumps(o)).output_map(structure=structure)['ediff']) - a.ediff) < a.ediff * 1e-2
  a.ediff = -1
  assert abs(a.ediff) < 1e-8
  assert abs(float(o.output_map(structure=structure)['ediff'])) < 1e-8

  a = Vasp()
  o = a._input['ediff_per_atom']
  d = {'EdiffPerAtom': o.__class__}
  assert a.ediff_per_atom is None
  assert a.ediff is None
  assert o.output_map() is None
  assert eval(repr(o), d).output_map() is None
  assert eval(repr(o), d).keyword == 'ediff'
  assert loads(dumps(o)).output_map() is None
  a.ediff = 1e-5
  a.ediff_per_atom = 2e-4
  assert abs(a.ediff_per_atom - 2e-4) < 1e-8
  assert a.ediff is None
  assert abs(float(o.output_map(structure=structure)['ediff']) - 2e-4 * N) < 2e-4 * 1e-2
  assert abs(float(eval(repr(o), d).output_map(structure=structure)['ediff']) - 2e-4 * N) < 2e-4 * 1e-2
  assert abs(float(loads(dumps(o)).output_map(structure=structure)['ediff']) - 2e-4 * N) < 2e-4 * 1e-2

  a.ediff = 1e-4
  a.ediff_per_atom = None
  assert abs(a.ediff-1e-4) < 1e-8
  assert a.ediff_per_atom is None
  a.ediff_per_atom = 1e-4
  a.ediff = None
  assert abs(a.ediff_per_atom-1e-4) < 1e-8
  assert a.ediff is None


def test_ediffg():
  from pickle import loads, dumps
  from pylada.vasp import Vasp
  from pylada.crystal import Structure
  a = Vasp()

  u = 0.25
  x, y = u, 0.25-u
  structure = Structure([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]) \
                       .add_atom(5.000000e-01, 5.000000e-01, 5.000000e-01, "Mg") \
                       .add_atom(5.000000e-01, 2.500000e-01, 2.500000e-01, "Mg") \
                       .add_atom(2.500000e-01, 5.000000e-01, 2.500000e-01, "Mg") \
                       .add_atom(2.500000e-01, 2.500000e-01, 5.000000e-01, "Mg") \
                       .add_atom(8.750000e-01, 8.750000e-01, 8.750000e-01, "Al") \
                       .add_atom(1.250000e-01, 1.250000e-01, 1.250000e-01, "Al") \
                       .add_atom(     x,     x,     x, "O") \
                       .add_atom(     x,     y,     y, "O") \
                       .add_atom(     y,     x,     y, "O") \
                       .add_atom(     y,     y,     x, "O") \
                       .add_atom(    -x,    -x,    -x, "O") \
                       .add_atom(    -x,    -y,    -y, "O") \
                       .add_atom(    -y,    -x,    -y, "O") \
                       .add_atom(    -y,    -y,    -x, "O") 
  
  N = float(len(structure))

  o = a._input['ediffg']
  d = {'Ediffg': o.__class__}
  assert a.ediffg is None
  assert a.ediffg_per_atom is None
  assert o.output_map() is None
  assert eval(repr(o), d).output_map() is None
  assert eval(repr(o), d).keyword == 'ediffg'
  assert loads(dumps(o)).output_map() is None
  a.ediffg_per_atom = 1e-5
  a.ediffg = 2e-4
  assert abs(a.ediffg - 2e-4) < 1e-8
  assert a.ediffg_per_atom is None
  assert abs(float(o.output_map(structure=structure)['ediffg']) - a.ediffg) < a.ediffg * 1e-2
  assert abs(float(eval(repr(o), d).output_map(structure=structure)['ediffg']) - a.ediffg) < a.ediffg * 1e-2
  assert abs(float(loads(dumps(o)).output_map(structure=structure)['ediffg']) - a.ediffg) < a.ediffg * 1e-2
  a.ediffg_per_atom = 1e-5
  a.ediffg = -2e-4
  assert abs(a.ediffg + 2e-4) < 1e-8
  assert a.ediffg_per_atom is None
  assert abs(float(o.output_map(structure=structure)['ediffg']) - a.ediffg) < -a.ediffg * 1e-2
  assert abs(float(eval(repr(o), d).output_map(structure=structure)['ediffg']) - a.ediffg) < -a.ediffg * 1e-2
  assert abs(float(loads(dumps(o)).output_map(structure=structure)['ediffg']) - a.ediffg) < -a.ediffg * 1e-2


  a.ediffg = 2e-4
  a.ediffg_per_atom = 2e-4
  o = a._input['ediffg_per_atom']
  d = {'EdiffgPerAtom': o.__class__}
  assert a.ediffg is None
  assert abs(a.ediffg_per_atom - 2e-4) < 1e-8
  assert abs(float(o.output_map(structure=structure)['ediffg']) - 2e-4*N) < 2e-4 * 1e-2
  assert abs(float(eval(repr(o), d).output_map(structure=structure)['ediffg']) - 2e-4*N) < 2e-4 * 1e-2
  assert abs(float(loads(dumps(o)).output_map(structure=structure)['ediffg']) - 2e-4*N) < 2e-4 * 1e-2
  a.ediffg = 2e-4
  a.ediffg_per_atom = -2e-4
  assert a.ediffg is None
  assert abs(a.ediffg_per_atom + 2e-4) < 1e-8
  assert abs(float(o.output_map(structure=structure)['ediffg']) + 2e-4) < 2e-4 * 1e-2
  assert abs(float(eval(repr(o), d).output_map(structure=structure)['ediffg']) + 2e-4) < 2e-4 * 1e-2
  assert abs(float(loads(dumps(o)).output_map(structure=structure)['ediffg']) + 2e-4) < 2e-4 * 1e-2
  a.ediffg_per_atom = None
  assert a.ediffg is None
  assert a.ediffg_per_atom is None
  assert o.output_map() is None
  assert eval(repr(o), d).output_map() is None
  assert eval(repr(o), d).keyword == 'ediffg'
  assert loads(dumps(o)).output_map() is None

  a.ediffg = 1e-4
  a.ediffg_per_atom = None
  assert abs(a.ediffg-1e-4) < 1e-8
  assert a.ediffg_per_atom is None
  a.ediffg_per_atom = 1e-4
  a.ediffg = None
  assert abs(a.ediffg_per_atom-1e-4) < 1e-8
  assert a.ediffg is None

if __name__ == '__main__':
  test_ediff()
  test_ediffg()
