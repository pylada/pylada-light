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

def test_istart():
  from time import sleep
  from collections import namedtuple
  from pickle import loads, dumps
  from os import remove, makedirs
  from os.path import join, exists
  from shutil import rmtree
  from tempfile import mkdtemp
  from pylada.vasp.files import WAVECAR
  from pylada.vasp import Vasp
  from pylada.misc import Changedir

  Extract = namedtuple("Extract", ['directory', 'success'])
  a = Vasp()
  o = a._input['istart']
  d = {'IStart': o.__class__}

  directory = mkdtemp()
  if directory in ['/tmp/test', '/tmp/test/']:
    if exists(directory): rmtree(directory)
    makedirs(directory)
  try: 
    assert a.istart == 'auto'
    assert o.output_map(vasp=a, outdir=directory)['istart'] == '0'
    assert eval(repr(o), d).value == 'auto'
    assert loads(dumps(o)).value == 'auto'

    restartdir = join(directory, 'restart')
    with Changedir(restartdir) as pwd: pass
    with open(join(restartdir, WAVECAR), 'w') as file: file.write('hello')

    # do not copy if not successful
    a.restart = Extract(restartdir, False)
    assert o.output_map(vasp=a, outdir=directory)['istart'] == '0'
    assert not exists(join(directory, 'WAVECAR'))

    # do not copy if file is empty
    a.restart = Extract(restartdir, True)
    with open(join(restartdir, WAVECAR), 'w') as file: pass
    assert o.output_map(vasp=a, outdir=directory)['istart'] == '0'
    assert not exists(join(directory, 'WAVECAR'))

    # now should copy
    assert a.istart == 'auto'
    with open(join(restartdir, WAVECAR), 'w') as file: file.write('hello')
    assert o.output_map(vasp=a, outdir=directory)['istart'] == '1'
    assert exists(join(directory, 'WAVECAR'))
      
    # check it copies only latest file.
    with open(join(restartdir, WAVECAR), 'w') as file: 
      file.write('hello')
      file.flush()
    with open(join(restartdir, WAVECAR), 'r') as file: pass
    sleep(1.5)
    with open(join(directory, WAVECAR), 'w') as file: file.write('hello world')
    with open(join(directory, WAVECAR), 'r') as file: pass
    assert o.output_map(vasp=a, outdir=directory)['istart'] == '1'
    assert exists(join(directory, 'WAVECAR'))
    with open(join(directory, WAVECAR), 'r') as file: 
      assert file.read().rstrip().lstrip() == 'hello world'
    
    sleep(0.2)
    with open(join(restartdir, WAVECAR), 'w') as file: file.write('hello')
    assert o.output_map(vasp=a, outdir=directory)['istart'] == '1'
    assert exists(join(directory, 'WAVECAR'))
    with open(join(directory, WAVECAR), 'r') as file: 
      assert file.read().rstrip().lstrip() == 'hello'

    # check if scratch is required
    remove(join(directory, WAVECAR))
    a.istart = 'scratch'
    assert a.istart == 'scratch'
    assert o.output_map(vasp=a, outdir=directory)['istart'] == '0'
    assert eval(repr(o), d).value == 'scratch'
    assert loads(dumps(o)).value == 'scratch'

  finally: 
    if directory not in ['/tmp/test', '/tmp/test/'] and exists(directory):
      rmtree(directory)


def test_icharg(): 
  from time import sleep
  from collections import namedtuple
  from pickle import loads, dumps
  from os import remove, makedirs
  from os.path import join, exists
  from shutil import rmtree
  from tempfile import mkdtemp
  from pylada.vasp.files import WAVECAR, CHGCAR
  from pylada.vasp import Vasp
  from pylada.misc import Changedir
  from pylada.error import ValueError

  Extract = namedtuple("Extract", ['directory', 'success'])
  a = Vasp()
  o = a._input['icharg']
  d = {'ICharg': o.__class__}

  directory = mkdtemp()
  if directory in ['/tmp/test', '/tmp/test/']:
    if exists(directory): rmtree(directory)
    makedirs(directory)
  try: 
    assert a.icharg == 'auto'
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '2'
    assert eval(repr(o), d).value == 'auto'
    assert loads(dumps(o)).value == 'auto'

    restartdir = join(directory, 'restart')
    with Changedir(restartdir) as pwd: pass
    with open(join(restartdir, CHGCAR), 'w') as file: file.write('hello')
    with open(join(restartdir, WAVECAR), 'w') as file: file.write('hello')

    # do not copy if not successful
    a.restart = Extract(restartdir, False)
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '2'
    assert not exists(join(directory, CHGCAR))
    assert not exists(join(directory, WAVECAR))

    # do not copy if empty
    with open(join(restartdir, CHGCAR), 'w') as file: pass
    with open(join(restartdir, WAVECAR), 'w') as file: pass
    a.restart = Extract(restartdir, True)
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '2'
    assert not exists(join(directory, CHGCAR))
    assert not exists(join(directory, WAVECAR))

    # now copy only CHGCAR
    with open(join(restartdir, CHGCAR), 'w') as file: file.write('hello')
    a.restart = Extract(restartdir, True)
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '1'
    assert exists(join(directory, CHGCAR))
    assert not exists(join(directory, WAVECAR))
    remove(join(directory, CHGCAR))
    # now copy only CHGCAR with scf
    a.nonscf = True
    a.restart = Extract(restartdir, True)
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '11'
    assert exists(join(directory, CHGCAR))
    assert not exists(join(directory, WAVECAR))
    remove(join(directory, CHGCAR))

    # now copy both 
    a.nonscf = False
    with open(join(restartdir, WAVECAR), 'w') as file: file.write('hello')
    a.restart = Extract(restartdir, True)
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '0'
    assert exists(join(directory, CHGCAR))
    assert exists(join(directory, WAVECAR))

    # now copy both with scf
    a.nonscf = True
    a.restart = Extract(restartdir, True)
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '10'
    assert exists(join(directory, CHGCAR))
    assert exists(join(directory, WAVECAR))

    # now check that latest is copied
    remove(join(restartdir, CHGCAR))
    remove(join(directory, CHGCAR))
    sleep(1.2)
    with open(join(directory, WAVECAR), 'w') as file: file.write('hello world')
    with open(join(directory, WAVECAR), 'r') as file: pass # Buffering issues..
    a.nonscf = False
    a.restart = Extract(restartdir, True)
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '0'
    assert exists(join(directory, WAVECAR))
    with open(join(directory, WAVECAR), 'r') as file:
      assert file.read().rstrip().lstrip() == 'hello world'
    with open(join(directory, WAVECAR), 'r') as file:
      assert file.read().rstrip().lstrip() != 'hello'

    with open(join(restartdir, WAVECAR), 'w') as file: file.write('hello')
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '0'
    assert exists(join(directory, WAVECAR))
    with open(join(directory, WAVECAR), 'r') as file:
      assert file.read().rstrip().lstrip() == 'hello'
    with open(join(directory, WAVECAR), 'r') as file:
      assert file.read().rstrip().lstrip() != 'hello world'

    # makes sure requests are honored
    # tries request. Should fail since CHGCAR does not exist.
    remove(join(directory, WAVECAR))
    a.icharg = 'chgcar'
    assert a.icharg == 'chgcar'
    try: o.output_map(vasp=a, outdir=directory)
    except ValueError: pass
    else: raise Exception()
    # now try for gold.
    with open(join(restartdir, CHGCAR), 'w') as file: file.write('hello')
    assert o.output_map(vasp=a, outdir=directory)['icharg'] == '1'
    assert exists(join(directory, CHGCAR))
    assert not exists(join(directory, WAVECAR))
    assert eval(repr(o), d).value == 'chgcar'
    assert loads(dumps(o)).value == 'chgcar'

  finally: 
    if directory not in ['/tmp/test', '/tmp/test/'] and exists(directory):
      rmtree(directory)

def test_istruc():
  from collections import namedtuple
  from pickle import loads, dumps
  from os import remove
  from os.path import join, exists
  from shutil import rmtree
  from tempfile import mkdtemp
  from pylada.vasp.files import POSCAR, CONTCAR
  from pylada.vasp import Vasp
  from pylada.crystal import Structure, read, specieset, write
  from pylada.error import ValueError

  structure = Structure([[0, 0.5, 0.5],[0.5, 0, 0.5], [0.5, 0.5, 0]], scale=5.43, name='has a name')\
                       .add_atom(0,0,0, "Si")\
                       .add_atom(0.25,0.25,0.25, "Si")

  Extract = namedtuple("Extract", ['directory', 'success', 'structure'])
  a = Vasp()
  o = a._input['istruc']
  d = {'IStruc': o.__class__}

  directory = mkdtemp()
  try: 
    assert a.istruc == 'auto'
    assert o.output_map(vasp=a, outdir=directory, structure=structure) is None
    assert eval(repr(o), d).value == 'auto'
    assert loads(dumps(o)).value == 'auto'
    assert exists(join(directory, POSCAR))
    remove(join(directory, POSCAR))

    # check reading from outcar but only on success.
    a.restart = Extract(directory, False, structure.copy())
    a.restart.structure[1].pos[0] += 0.02
    assert a.istruc == 'auto'
    assert o.output_map(vasp=a, outdir=directory, structure=structure) is None
    assert exists(join(directory, POSCAR))
    other = read.poscar(join(directory, POSCAR), types=specieset(structure))
    assert abs(other[1].pos[0] - 0.25) < 1e-8
    assert abs(other[1].pos[0] - 0.27) > 1e-8
    # check reading from outcar but only on success.
    a.restart = Extract(directory, True, structure.copy())
    a.restart.structure[1].pos[0] += 0.02
    assert a.istruc == 'auto'
    assert o.output_map(vasp=a, outdir=directory, structure=structure) is None
    assert exists(join(directory, POSCAR))
    other = read.poscar(join(directory, POSCAR), types=specieset(structure))
    assert abs(other[1].pos[0] - 0.25) > 1e-8
    assert abs(other[1].pos[0] - 0.27) < 1e-8

    # Now check CONTCAR
    write.poscar(structure, join(directory, CONTCAR))
    assert a.istruc == 'auto'
    assert o.output_map(vasp=a, outdir=directory, structure=structure) is None
    assert exists(join(directory, POSCAR))
    other = read.poscar(join(directory, POSCAR), types=specieset(structure))
    assert abs(other[1].pos[0] - 0.25) < 1e-8
    assert abs(other[1].pos[0] - 0.27) > 1e-8

    # Check some failure modes.
    write.poscar(structure, join(directory, CONTCAR))
    structure[0].type = 'Ge'
    a.restart = None
    try: o.output_map(vasp=a, outdir=directory, structure=structure)
    except ValueError: pass
    else: raise Exception()
    structure[0].type = 'Si'
    structure.add_atom(0.25,0,0, 'Si')
    try: o.output_map(vasp=a, outdir=directory, structure=structure)
    except ValueError: pass
    else: raise Exception()

  finally: rmtree(directory)


if __name__ == '__main__':
  test_istart()
  test_icharg()
  test_istruc()
