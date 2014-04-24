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

def test(Class, withposcar):
  from pickle import loads, dumps
  from os import chdir, getcwd, remove
  from os.path import join, exists
  from shutil import rmtree
  from tempfile import mkdtemp
  from pylada.vasp.files import WAVECAR, CHGCAR, CONTCAR, POSCAR
  from restart_class import Extract, Vasp

  cwd = getcwd()
  directory = mkdtemp()
  try: 
    # no prior run.
    for v, istart, icharg in [(Vasp(True), None, 12), (Vasp(False), None, None)]:
      playdir = mkdtemp()
      try: 
        chdir(playdir)
        v.istart, v.icharge = None, None
        assert Class(None).incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert not exists(join(playdir, CHGCAR))
        assert not exists(join(playdir, WAVECAR))
        assert not exists(join(playdir, POSCAR))

        v.istart, v.icharge = None, None
        r = loads(dumps(Class(None)))
        assert r.incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert not exists(join(playdir, CHGCAR))
        assert not exists(join(playdir, WAVECAR))
        assert not exists(join(playdir, POSCAR))
      finally:
        chdir(cwd)
        rmtree(playdir)

    # no prior run.
    for v, istart, icharg in [(Vasp(True), None, 12), (Vasp(False), None, None)]:
      playdir = mkdtemp()
      try: 
        chdir(playdir)
        v.istart, v.icharge = None, None
        assert Class(Extract(directory, False)).incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert not exists(join(playdir, CHGCAR))
        assert not exists(join(playdir, WAVECAR))
        assert not exists(join(playdir, POSCAR))

        v.istart, v.icharge = None, None
        r = loads(dumps(Class(Extract(directory, False))))
        assert r.incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert r.value.directory == directory 
        assert r.value.success == False
        assert not exists(join(playdir, CHGCAR))
        assert not exists(join(playdir, WAVECAR))
        assert not exists(join(playdir, POSCAR))
      finally:
        chdir(cwd)
        rmtree(playdir)
    
    # empty prior run.
    with open(join(directory, CHGCAR), "w") as file: pass
    with open(join(directory, WAVECAR), "w") as file: pass
    with open(join(directory, CONTCAR), "w") as file: pass
    for v, istart, icharg in [(Vasp(True), 0, 12), (Vasp(False), 0, 2)]:
      playdir = mkdtemp()
      try: 
        chdir(playdir)
        v.istart, v.icharge = None, None
        assert Class(Extract(directory, True)).incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert not exists(join(playdir, CHGCAR))
        assert not exists(join(playdir, WAVECAR))
        assert not exists(join(playdir, POSCAR))

        v.istart, v.icharge = None, None
        r = loads(dumps(Class(Extract(directory, True))))
        assert r.incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert r.value.directory == directory 
        assert r.value.success == True
        assert not exists(join(playdir, CHGCAR))
        assert not exists(join(playdir, WAVECAR))
        assert not exists(join(playdir, POSCAR))
      finally:
        chdir(cwd)
        rmtree(playdir)
    
    # prior run with charge and contcar only.
    with open(join(directory, CHGCAR), "w") as file: file.write('hello')
    with open(join(directory, CONTCAR), "w") as file: file.write('hello')
    for v, istart, icharg in [(Vasp(True), 0, 11), (Vasp(False), 0, 1)]:
      playdir = mkdtemp()
      try: 
        chdir(playdir)
        v.istart, v.icharge = None, None
        assert Class(Extract(directory, True)).incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert exists(join(playdir, CHGCAR))
        assert not exists(join(playdir, WAVECAR))
        if withposcar:
          assert exists(join(playdir, POSCAR))
          remove(join(playdir, POSCAR))
        else: assert not exists(join(playdir, POSCAR))
        remove(join(playdir, CHGCAR))

        v.istart, v.icharge = None, None
        r = loads(dumps(Class(Extract(directory, True))))
        assert r.incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert r.value.directory == directory 
        assert r.value.success == True
        assert exists(join(playdir, CHGCAR))
        assert not exists(join(playdir, WAVECAR))
        if withposcar:
          assert exists(join(playdir, POSCAR))
          remove(join(playdir, POSCAR))
        else: assert not exists(join(playdir, POSCAR))
      finally:
        chdir(cwd)
        rmtree(playdir)

    # prior run with charge and wavecar.
    with open(join(directory, WAVECAR), "w") as file: file.write('hello')
    for v, istart, icharg in [(Vasp(True), 1, 11), (Vasp(False), 1, 1)]:
      playdir = mkdtemp()
      try: 
        chdir(playdir)
        v.istart, v.icharge = None, None
        assert Class(Extract(directory, True)).incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert exists(join(playdir, CHGCAR))
        assert exists(join(playdir, WAVECAR))
        if withposcar:
          assert exists(join(playdir, POSCAR))
          remove(join(playdir, POSCAR))
        else: assert not exists(join(playdir, POSCAR))
        remove(join(playdir, CHGCAR))
        remove(join(playdir, WAVECAR))

        v.istart, v.icharge = None, None
        r = loads(dumps(Class(Extract(directory, True))))
        assert r.incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert r.value.directory == directory 
        assert r.value.success == True
        assert exists(join(playdir, CHGCAR))
        assert exists(join(playdir, WAVECAR))
        if withposcar:
          assert exists(join(playdir, POSCAR))
          remove(join(playdir, POSCAR))
        else: assert not exists(join(playdir, POSCAR))
      finally:
        chdir(cwd)
        rmtree(playdir)

    # prior run with wavecar only.
    remove(join(directory, CHGCAR))
    with open(join(directory, CHGCAR), "w") as file: pass
    with open(join(directory, CONTCAR), "w") as file: pass
    for v, istart, icharg in [(Vasp(True), 1, 10), (Vasp(False), 1, 0)]:
      playdir = mkdtemp()
      try: 
        chdir(playdir)
        v.istart, v.icharge = None, None
        assert Class(Extract(directory, True)).incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert not exists(join(playdir, CHGCAR))
        assert exists(join(playdir, WAVECAR))
        assert not exists(join(playdir, POSCAR))
        remove(join(playdir, WAVECAR))

        v.istart, v.icharge = None, None
        r = loads(dumps(Class(Extract(directory, True))))
        assert r.incar_string(vasp=v) is None
        assert v.istart == istart and v.icharg == icharg
        assert r.value.directory == directory 
        assert r.value.success == True
        assert not exists(join(playdir, CHGCAR))
        assert exists(join(playdir, WAVECAR))
        assert not exists(join(playdir, POSCAR))
      finally:
        chdir(cwd)
        rmtree(playdir)

  finally: rmtree(directory)

if __name__ == "__main__":
  from sys import argv, path 
  from numpy import array
  from pylada.vasp.incar._params import Restart, PartialRestart
  if len(argv) > 0: path.extend(argv[1:])
  
  test(PartialRestart, False)
  test(Restart, True)

