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

def test(executable):
  """ Tests IteratorProcess. Includes failure modes.  """
  from tempfile import mkdtemp
  from shutil import rmtree
  from numpy import all, arange, abs, array
  from pylada.process.iterator import IteratorProcess
  from pylada.process import Fail, NotStarted
  from pylada import default_comm
  from functional import Functional
  comm = default_comm.copy()
  dir = mkdtemp()
  try: 
    functional = Functional(executable, range(8))
    program = IteratorProcess(functional, outdir=dir)
    # program not started. should fail.
    try: program.poll()
    except NotStarted: pass
    else: raise Exception()
    try: program.wait()
    except NotStarted: pass
    else: raise Exception()

    # now starting for real.
    program.start(comm)
    while not program.poll():  continue
    extract = functional.Extract(dir)
    assert extract.success
    assert all(arange(8) - extract.order == 0)
    assert all(abs(extract.pi - array([0.0, 3.2, 3.162353, 3.150849,
                                       3.146801, 3.144926, 3.143907, 3.143293]))\
                < 1e-5 )
    assert all(abs(extract.error - array([3.141593, 0.05840735, 0.02076029, 0.009256556,
                                          0.005207865, 0.00333321, 0.002314774, 0.001700664]))\
                < 1e-5 )
    assert all(n['n'] == comm['n'] for n in extract.comm)
    # restart
    assert program.poll()
    program.start(comm)
    assert program.process is None
    assert program.poll()
    # true restart
    program = IteratorProcess(functional, outdir=dir)
    program.start(comm)
    assert program.process is None
    assert program.poll()
    extract = functional.Extract(dir)
    assert extract.success
    assert all(arange(8) - extract.order == 0)
    assert all(abs(extract.pi - array([0.0, 3.2, 3.162353, 3.150849,
                                       3.146801, 3.144926, 3.143907, 3.143293]))\
                < 1e-5 )
    assert all(abs(extract.error - array([3.141593, 0.05840735, 0.02076029, 0.009256556,
                                          0.005207865, 0.00333321, 0.002314774, 0.001700664]))\
                < 1e-5 )
  finally:
    try: rmtree(dir)
    except: pass

  try: 
    functional = Functional(executable, [666])
    program = IteratorProcess(functional, outdir=dir)
    program.start(comm)
    program.wait()
  except Fail: pass
  else: raise Exception
  finally:
    try: rmtree(dir)
    except: pass
  try: 
    functional = Functional(executable, [667])
    program = IteratorProcess(functional, outdir=dir)
    program.start(comm)
    program.wait()
  finally:
    try: rmtree(dir)
    except: pass

if __name__ == "__main__":
  from sys import argv, path
  from os.path import abspath
  if len(argv) < 1: raise ValueError("test need to be passed location of pifunc.")
  if len(argv) > 2: path.extend(argv[2:])
  test(abspath(argv[1]))
