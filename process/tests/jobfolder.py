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
  """ Tests JobFolderProcess. Includes failure modes.  """
  from tempfile import mkdtemp
  from os.path import join
  from shutil import rmtree
  from numpy import all, arange, abs, array
  from pylada.jobfolder.jobfolder import JobFolder
  from pylada.jobfolder.massextract import MassExtract
  from pylada.jobfolder import save
  from pylada.process.jobfolder import JobFolderProcess
  from pylada.process import Fail, AlreadyStarted, NotStarted
  from pylada import default_comm
  from functional import Functional

  root = JobFolder()
  for n in xrange(8):
    job = root / str(n)
    job.functional = Functional(executable, [n])
    job.params['sleep'] = 1

  comm = default_comm.copy()
  comm['n'] = 4

  dir = mkdtemp()
  try: 
    program = JobFolderProcess(root, nbpools=2, outdir=dir)
    assert program.nbjobsleft > 0
    # program not started. should fail.
    try: program.poll()
    except NotStarted: pass
    else: raise Exception()
    try: program.wait()
    except NotStarted: pass
    else: raise Exception()

    # now starting for real.
    program.start(comm)
    assert len(program.process) == 2
    # Should not be possible to start twice.
    try: program.start(comm)
    except AlreadyStarted: pass
    else: raise Exception()
    while not program.poll():  continue
    assert program.nbjobsleft == 0
    save(root, join(dir, 'dict.dict'), overwrite=True)
    extract = MassExtract(join(dir, 'dict.dict'))
    assert all(extract.success.itervalues())
    order = array(extract.order.values()).flatten()
    assert all(arange(8) - order == 0)
    pi = array(extract.pi.values()).flatten()
    assert all(abs(pi - array([0.0, 3.2, 3.162353, 3.150849,
                               3.146801, 3.144926, 3.143907, 3.143293]))\
                < 1e-5 )
    error = array(extract.error.values()).flatten()
    assert all(abs(error - array([3.141593, 0.05840735, 0.02076029, 0.009256556,
                                  0.005207865, 0.00333321, 0.002314774, 0.001700664]))\
                < 1e-5 )
    assert all(n['n'] == comm['n'] for n in extract.comm)
    # restart
    assert program.poll()
    assert len(program.process) == 0
    program.start(comm)
    assert len(program.process) == 0
    assert program.poll()
  finally:
    try: rmtree(dir)
    except: pass

  try: 
    job = root / str(666)
    job.functional = Functional(executable, [666])
    program = JobFolderProcess(root, nbpools=2, outdir=dir)
    assert program.nbjobsleft > 0
    program.start(comm)
    program.wait()
    assert program.nbjobsleft == 0
  except Fail as r: 
    assert len(program.errors.keys()) == 1
    assert '666' in program.errors
    assert len(program._finished) == 8
  else: raise Exception
  finally:
    try: rmtree(dir)
    except: pass
  try: 
    job.functional.order = [667]
    program = JobFolderProcess(root, nbpools=2, outdir=dir)
    assert program.nbjobsleft > 0
    program.start(comm)
    program.wait()
    assert program.nbjobsleft == 0
  finally:
    try: rmtree(dir)
    except: pass


def test_update(executable):
  """ Tests JobFolderProcess with update. """
  from tempfile import mkdtemp
  from shutil import rmtree
  from pylada.jobfolder.jobfolder import JobFolder
  from pylada.process.jobfolder import JobFolderProcess
  from pylada import default_comm
  from functional import Functional

  root = JobFolder()
  for n in xrange(3):
    job = root / str(n)
    job.functional = Functional(executable, [n])
    job.params['sleep'] = 1
  supp = JobFolder()
  for n in xrange(3, 6):
    job = supp / str(n)
    job.functional = Functional(executable, [n])
    job.params['sleep'] = 1

  comm = default_comm.copy()
  comm['n'] = 4

  dir = mkdtemp()
  try: 
    program = JobFolderProcess(root, nbpools=2, outdir=dir, keepalive=True)
    assert program.keepalive 

    # compute current jobs.
    program.start(comm)
    program.wait()
    assert hasattr(program, '_comm')

    # compute second set of updated jobs
    program.update(supp)
    program.wait()

  finally:
    try: rmtree(dir)
    except: pass

  # check with deleteold=True
  dir = mkdtemp()
  try: 
    program = JobFolderProcess(root, nbpools=2, outdir=dir, keepalive=True)
    assert program.keepalive 

    # compute current jobs.
    program.start(comm)
    program.wait()
    assert hasattr(program, '_comm')

    # compute second set of updated jobs
    program.update(supp, deleteold=True)
    assert hasattr(program, '_comm')
    program.wait()

  finally:
    try: rmtree(dir)
    except: pass

def test_update_with_fail(executable):
  """ Tests JobFolderProcess with update. """
  from tempfile import mkdtemp
  from shutil import rmtree
  from pylada.jobfolder.jobfolder import JobFolder
  from pylada.process.jobfolder import JobFolderProcess
  from pylada.process import Fail
  from pylada import default_comm
  from functional import Functional

  root = JobFolder()
  for n in xrange(3):
    job = root / str(n)
    job.functional = Functional(executable, [n])
    job.params['sleep'] = 1
  root['1'].functional.order = 666
  root['1'].sleep = None
  supp = JobFolder()
  for n in xrange(3, 6):
    job = supp / str(n)
    job.functional = Functional(executable, [n])
    job.params['sleep'] = 1
  supp['5'].sleep = 0
  supp['5'].functional.order = 666

  comm = default_comm.copy()
  comm['n'] = 4

  dir = mkdtemp()
  try: 
    program = JobFolderProcess(root, nbpools=2, outdir=dir, keepalive=True)

    # compute current jobs.
    program.start(comm)
    try: program.wait()
    except Fail: pass
    else: raise Exception()
    assert len(program.errors) == 1

    # compute second set of updated jobs
    program.update(supp)
    try: program.wait()
    except Fail: pass
    else: raise Exception()
    assert len(program.errors) == 2
    program.errors.clear()


  finally:
    try: rmtree(dir)
    except: pass

if __name__ == "__main__":
  from sys import argv, path
  from os.path import abspath
  if len(argv) < 1: raise ValueError("test need to be passed location of pifunc.")
  if len(argv) > 2: path.extend(argv[2:])
  test(abspath(argv[1]))
  test_update(abspath(argv[1]))
  test_update_with_fail(abspath(argv[1]))
