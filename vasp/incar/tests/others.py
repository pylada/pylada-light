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

def test_choices():
  from pickle import loads, dumps
  from pylada.vasp.incar._params import Precision, PrecFock, IniWave

  assert PrecFock(None).incar_string() is None
  for i in ['Low', 'Medium', 'Accurate', 'Fast', 'Normal']:
    assert PrecFock(i).incar_string() == 'PRECFOCK = {0}'.format(i)
    assert loads(dumps(PrecFock(i))).incar_string() == 'PRECFOCK = {0}'.format(i)
    assert repr(PrecFock(i)) == "PrecFock({0!r})".format(i)
  try: PrecFock('ssdfds')
  except: pass
  else: raise RuntimeError()

  assert Precision(None).incar_string() is None
  for i in ['Low', 'Medium', 'High', 'Accurate', 'Normal', 'Single']:
    assert Precision(i).incar_string() == 'PREC = {0}'.format(i)
    assert loads(dumps(Precision(i))).incar_string() == 'PREC = {0}'.format(i)
    assert repr(Precision(i)) == "Precision({0!r})".format(i)
  try: Precision('ssdfds')
  except: pass
  else: raise RuntimeError()

  assert IniWave(None).incar_string() is None
  for i, j in [(0, 'jellium'), (1, 'random')]:
    assert IniWave(i).incar_string() == 'INIWAV = {0}'.format(i)
    assert IniWave(j).incar_string() == 'INIWAV = {0}'.format(i)
    assert loads(dumps(IniWave(i))).incar_string() == 'INIWAV = {0}'.format(i)
    assert loads(dumps(IniWave(j))).incar_string() == 'INIWAV = {0}'.format(i)
    assert repr(IniWave(i)) == "IniWave({0!r})".format(j)
  try: IniWave('ssdfds')
  except: pass
  else: raise RuntimeError()
  try: IniWave(3)
  except: pass
  else: raise RuntimeError()

def test_ediff():
  from pickle import loads, dumps
  from pylada.crystal.cppwrappers import Structure, supercell
  from pylada.vasp.incar._params import Ediff, Ediffg

  structure = Structure([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])\
                       .add_atom(0, 0, 0, 'Si')\
                       .add_atom(0.25, 0.25, 0.25, 'Si')
  assert Ediff(None).incar_string(structure=structure) is None
  a = loads(dumps(Ediff(1e-4))).incar_string(structure=structure).split()
  assert a[0] == 'EDIFF' and a[1] == '=' and abs(float(a[2]) - 1e-4 * 2.) < 1e-8
  a = loads(dumps(Ediff(-1e-4))).incar_string(structure=structure).split() 
  assert a[0] == 'EDIFF' and a[1] == '=' and abs(float(a[2]) - 1e-4) < 1e-8 and float(a[2]) > 0e0
  
  assert Ediffg(None).incar_string(structure=structure) is None
  a = loads(dumps(Ediffg(1e-4))).incar_string(structure=structure).split()
  assert a[0] == 'EDIFFG' and a[1] == '=' and abs(float(a[2]) - 1e-4*2.) < 1e-8
  a = loads(dumps(Ediffg(-1e-4))).incar_string(structure=structure).split()
  assert a[0] == 'EDIFFG' and a[1] == '=' and abs(float(a[2]) + 1e-4) < 1e-8

  structure = supercell(structure, [[1,0,0],[0,1,0],[0,0,1]])
  a = loads(dumps(Ediff(1e-4))).incar_string(structure=structure).split()
  assert a[0] == 'EDIFF' and a[1] == '=' and abs(float(a[2]) - 1e-4 * 8.) < 1e-8
  a = loads(dumps(Ediff(-1e-4))).incar_string(structure=structure).split() 
  assert a[0] == 'EDIFF' and a[1] == '=' and abs(float(a[2]) - 1e-4) < 1e-8 and float(a[2]) > 0e0
  
  a = loads(dumps(Ediffg(1e-4))).incar_string(structure=structure).split()
  assert a[0] == 'EDIFFG' and a[1] == '=' and abs(float(a[2]) - 1e-4*8.) < 1e-8
  a = loads(dumps(Ediffg(-1e-4))).incar_string(structure=structure).split()
  assert a[0] == 'EDIFFG' and a[1] == '=' and abs(float(a[2]) + 1e-4) < 1e-8


def test_encut(Encut):
  from pickle import loads, dumps
  from collections import namedtuple
  from pylada.crystal.cppwrappers import Structure, supercell
  from quantities import eV, hartree

  Vasp = namedtuple('Vasp', ['species'])
  Specie = namedtuple('Specie', ['enmax'])
  vasp = Vasp({'Si': Specie(1.*eV), 'Ge': Specie(10.), 'C': Specie(100.*eV)})
  name = Encut.__name__.upper()

  structure = Structure([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])\
                       .add_atom(0, 0, 0, 'Si')\
                       .add_atom(0.25, 0.25, 0.25, 'Ge')
  assert Encut(None).incar_string(vasp=vasp, structure=structure) is None
  a = loads(dumps(Encut(50))).incar_string(vasp=vasp, structure=structure).split()
  assert a[0] == name and a[1] == '=' and abs(float(a[2]) - 50) < 1e-8
  a = loads(dumps(Encut(1.0))).incar_string(vasp=vasp, structure=structure).split() 
  assert a[0] == name and a[1] == '=' and abs(float(a[2]) - 10.) < 1e-8 
  structure[0].type = 'C'
  a = loads(dumps(Encut(2.0))).incar_string(vasp=vasp, structure=structure).split() 
  assert a[0] == name and a[1] == '=' and abs(float(a[2]) - 2.*100.) < 1e-8 
  a = loads(dumps(Encut(50. * eV))).incar_string(vasp=vasp, structure=structure).split() 
  assert a[0] == name and a[1] == '=' and abs(float(a[2]) - 50.) < 1e-8 
  a = loads(dumps(Encut((50. * eV).rescale(hartree)))).incar_string(vasp=vasp, structure=structure).split() 
  assert a[0] == name and a[1] == '=' and abs(float(a[2]) - 50.) < 1e-8 
  assert Encut(-50).incar_string(vasp=vasp, structure=structure) is None
  assert Encut(-50*eV).incar_string(vasp=vasp, structure=structure) is None

if __name__ == "__main__":
  from sys import argv, path 
  from numpy import array
  if len(argv) > 0: path.extend(argv[1:])
  from pylada.vasp.incar._params import Encut, EncutGW
  
  test_choices()
  test_ediff()
  test_encut(Encut)
  test_encut(EncutGW)

