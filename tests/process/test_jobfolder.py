###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to
#  make it easier to submit large numbers of jobs on supercomputers. It
#  provides a python interface to physical input, such as crystal structures,
#  as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs.
#  It is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License along with
#  PyLaDa.  If not, see <http://www.gnu.org/licenses/>.
###############################
from pytest import fixture, mark

from .conftest import jobfolders, mpi4py_required, FakeFunctional


@fixture
def do_multiple_mpi_programs():
    import pylada
    old = pylada.do_multiple_mpi_programs
    pylada.do_multiple_mpi_programs = True
    yield True
    pylada.do_multiple_mpi_programs = False


@fixture
def root(executable):
    return jobfolders(executable, 0, 8)


def processalloc(job):
    """returns a random number between 1 and 4 included."""
    from random import randint
    return randint(1, 4)


def jobfolder_process(tmpdir, root, **kwargs):
    from pylada.process.jobfolder import JobFolderProcess
    return JobFolderProcess(root, nbpools=2, outdir=str(tmpdir), **kwargs)


def pool_process(tmpdir, root, pal=processalloc, **kwargs):
    from pylada.process.pool import PoolProcess
    return PoolProcess(root, processalloc=pal, outdir=str(tmpdir), **kwargs)


@fixture(params=[jobfolder_process, pool_process])
def program(request, tmpdir, root):
    return request.param(tmpdir, root)


def test_cannot_poll_before_starting(program):
    from pytest import raises
    from pylada.process import NotStarted
    assert program.nbjobsleft > 0
    # program not started. should fail.
    with raises(NotStarted):
        program.poll()

    with raises(NotStarted):
        program.wait()


@mpi4py_required
def test_cannot_start_twice(tmpdir, root, comm, do_multiple_mpi_programs):
    from pytest import raises
    from pylada.process import AlreadyStarted
    program = jobfolder_process(tmpdir, root)
    program.start(comm)
    assert len(program.process) == 2
    # Should not be possible to start twice.
    with raises(AlreadyStarted):
        program.start(comm)

    program.kill()


@mpi4py_required
def test_completes(program, comm, tmpdir, root, do_multiple_mpi_programs):
    from numpy import all, arange, abs, array
    from pylada.jobfolder.massextract import MassExtract
    from pylada.jobfolder import save

    program.start(comm)
    program.wait()
    assert program.nbjobsleft == 0

    save(root, str(tmpdir.join('dict.dict')), overwrite=True)
    extract = MassExtract(str(tmpdir.join('dict.dict')))
    assert all(extract.success.values())
    order = array(list(extract.order.values())).flatten()
    assert all(arange(8) - order == 0)
    pi = array(list(extract.pi.values())).flatten()
    expected = [
        0.0, 3.2, 3.162353, 3.150849, 3.146801, 3.144926, 3.143907, 3.143293
    ]
    assert all(abs(pi - array(expected)) < 1e-5)
    error = array(list(extract.error.values())).flatten()
    expected = [
        3.141593, 0.05840735, 0.02076029, 0.009256556, 0.005207865, 0.00333321,
        0.002314774, 0.001700664
    ]
    assert all(abs(error - array(expected)) < 1e-5)
    assert all(n['n'] == comm['n'] for n in extract.comm)


@mpi4py_required
def test_restart(program, comm, do_multiple_mpi_programs):
    program.start(comm)
    program.wait()
    assert program.poll()
    assert len(program.process) == 0
    program.start(comm)
    assert len(program.process) == 0
    assert program.poll()


@mpi4py_required
@mark.parametrize('Process', [jobfolder_process, pool_process])
def test_failed_job_discovery(tmpdir, comm, root, executable, Process,
                              do_multiple_mpi_programs):
    """Tests JobFolderProcess.

    Includes failure modes.
    """
    from pytest import raises
    from pylada.process import Fail

    with raises(Fail):
        job = root / str(666)
        job.functional = FakeFunctional(executable, [50], fail='end')
        program = Process(tmpdir, root)
        assert program.nbjobsleft > 0
        program.start(comm)
        program.wait()
        assert program.nbjobsleft == 0

    assert len(program.errors.keys()) == 1
    assert '666' in program.errors
    assert len(program._finished) == 8


@mpi4py_required
@mark.parametrize('Process', [jobfolder_process, pool_process])
def test_restart_failed_job(tmpdir, comm, root, executable, Process,
                            do_multiple_mpi_programs):
    from pytest import raises
    from pylada.process import Fail

    with raises(Fail):
        job = root / str(666)
        job.functional = FakeFunctional(executable, [50], fail='end')
        program = Process(tmpdir, root)
        program.start(comm)
        program.wait()

    job.functional.order = [45]
    job.functional.fail = None
    program = Process(tmpdir, root)
    assert program.nbjobsleft > 0
    program.start(comm)
    program.wait()
    assert program.nbjobsleft == 0


@mpi4py_required
@mark.parametrize('Process', [jobfolder_process, pool_process])
def test_update(tmpdir, executable, comm, Process, do_multiple_mpi_programs):
    from copy import deepcopy
    from pylada.jobfolder.massextract import MassExtract
    from pylada.jobfolder import save

    # creates and start computing first set of jobs
    root = jobfolders(executable, 0, 3)
    program = Process(tmpdir, deepcopy(root), keepalive=True)
    assert program.keepalive
    program.start(comm)
    assert hasattr(program, '_comm')

    # update jobfolder and start computing second set of jobs
    supp = jobfolders(executable, 3, 6)
    program.update(supp)
    for key in supp.keys():
        assert key in program.jobfolder
    for key in root.keys():
        assert key in program.jobfolder
    assert len(program.jobfolder) == len(root) + len(supp)

    # create mass extraction object and check succcess
    save(program.jobfolder, str(tmpdir.join('dict.dict')), overwrite=True)
    extract = MassExtract(str(tmpdir.join('dict.dict')))

    # wait for job completion and check for success
    program.wait()
    assert len(extract.success.values()) == 6
    assert all((u for u in extract.success.values()))


@mpi4py_required
@mark.parametrize('Process', [jobfolder_process, pool_process])
def test_update_with_deleteold(tmpdir, executable, comm, Process,
                               do_multiple_mpi_programs):
    # creates and start computing first set of jobs
    root = jobfolders(executable, 0, 3)
    program = Process(tmpdir, root, keepalive=True)
    program.start(comm)

    supp = jobfolders(executable, 3, 6)
    # wait for completion of current jobs, check that update with delete jobs results in only
    # uncompleted jobs
    program.wait()
    program.update(supp, deleteold=True)
    for key in supp.keys():
        assert key in program.jobfolder
    assert len(program.jobfolder) == len(supp)


@mpi4py_required
@mark.parametrize('Process', [jobfolder_process, pool_process])
def test_update_with_fail(tmpdir, executable, comm, Process,
                          do_multiple_mpi_programs):
    """Tests JobFolderProcess with update and failure."""
    from pytest import raises
    from pylada.process import Fail
    from pylada.jobfolder.massextract import MassExtract
    from pylada.jobfolder import save

    root = jobfolders(executable, 0, 3)
    root['1'].functional.order = 68
    root['1'].functional.fail = 'end'
    root['1'].sleep = None

    supp = jobfolders(executable, 3, 6)
    supp['5'].sleep = 0
    supp['5'].functional.order = 78
    supp['5'].functional.fail = 'midway'

    program = Process(tmpdir, root, keepalive=True)

    # compute current jobs.
    program.start(comm)
    with raises(Fail):
        program.wait()
    assert len(program.errors) == 1

    # compute second set of updated jobs
    program.update(supp)
    with raises(Fail):
        program.wait()
    assert len(program.errors) == 2
