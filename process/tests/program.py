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
  """ Tests ProgramProcess. Includes failure modes.  """
  from tempfile import mkdtemp
  from os.path import join 
  from shutil import rmtree
  from pylada.process.program import ProgramProcess
  from pylada.process import Fail, NotStarted
  from pylada.misc import Changedir
  from pylada import default_comm as comm
  from functional import ExtractSingle as Extract
  dir = mkdtemp()
  try: 
    with Changedir(dir) as pwd: pass
    stdout = join(dir, 'stdout')
    program = ProgramProcess( executable, outdir=dir, 
                              cmdline=['--sleep', 0, '--order', 4], 
                              stdout=stdout, dompi=True )
    # program not started. should fail.
    try: program.poll()
    except NotStarted: pass
    else: raise Exception()
    try: program.wait()
    except NotStarted: pass
    else: raise Exception()

    # now starting for real.
    assert program.start(comm) == False
    assert program.process is not None
    while not program.poll():  continue
    extract = Extract(stdout)
    assert extract.success
    assert abs(extract.pi-3.146801e+00) < 1e-2 * extract.error
    assert abs(extract.error-0.005207865) < 1e-2 * extract.error
    assert extract.comm['n'] == comm['n']
    # restart
    assert program.process is None
    program.start(comm)
    assert program.process is None
  finally: rmtree(dir)

  # fail on poll
  try: 
    with Changedir(dir) as pwd: pass
    stdout = join(dir, 'stdout')
    program = ProgramProcess( executable, outdir=dir, 
                              stderr=join(dir, 'shit'),
                              cmdline=['--sleep', 0, '--order', 666], 
                              stdout=stdout, dompi=True )
    program.start(comm)
    while not program.poll():  continue
  except Fail: pass
  except: raise
  else: raise Exception()
  finally: rmtree(dir)

  # fail on wait
  try: 
    with Changedir(dir) as pwd: pass
    stdout = join(dir, 'stdout')
    program = ProgramProcess( executable, outdir=dir, 
                              stderr=join(dir, 'shit'),
                              cmdline=['--sleep', 0, '--order', 666], 
                              stdout=stdout, dompi=True )
    program.start(comm)
    program.wait()
  except Fail: pass
  else: raise Exception()
  finally: rmtree(dir)

  try: 
    with Changedir(dir) as pwd: pass
    stdout = join(dir, 'stdout')
    program = ProgramProcess( executable, outdir=dir, 
                              stderr=join(dir, 'shit'),
                              cmdline=['--sleep', 0, '--order', 6666], 
                              stdout=stdout, dompi=True )
    program.start(comm)
    while not program.poll():  continue
  except Fail: pass
  else: raise Exception()
  finally: rmtree(dir)

if __name__ == "__main__":
  from sys import argv, path
  from os.path import abspath
  if len(argv) < 1: raise ValueError("test need to be passed location of pifunc.")
  if len(argv) > 2: path.extend(argv[2:])
  test(abspath(argv[1]))
