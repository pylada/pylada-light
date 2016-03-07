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
from fixtures import executable, jobfolders
from pytest import mark, fixture
from pylada.process.pool import PoolProcess


@fixture
def comm():
    from pylada import default_comm
    result = default_comm.copy()
    result['n'] = 4
    return result


def processalloc(job):
    """ returns a random number between 1 and 4 included. """
    from random import randint
    return randint(1, 4)


def test_failures(tmpdir, executable, comm):
    """ Tests whether scheduling jobs works on known failure cases. """
    from pylada import default_comm
    from functional import Functional
    root = jobfolders(executable, 0, 8)

    def processalloc_test1(job):
        d = {'1': 1, '0': 3, '3': 3, '2': 3, '5': 3, '4': 2, '7': 2, '6': 1}
        return d[job.name[1:-1]]

    program = PoolProcess(root, processalloc=processalloc_test1,
                          outdir=str(tmpdir))
    program._comm = comm
    for i in xrange(10000):
        jobs = program._getjobs()
        assert sum(program._alloc[u] for u in jobs) <= program._comm['n'],\
            (jobs, [program._alloc[u] for u in jobs])


@mark.parametrize('nprocs, njobs', [(8, 20), (16, 20)])
def test_getjobs(comm, tmpdir, executable, nprocs, njobs):
    """ Test scheduling. """
    root = jobfolders(executable, 0, 8)

    def processalloc(job):
        """ returns a random number between 1 and 4 included. """
        from random import randint
        return randint(1, comm['n'])

    for j in xrange(100):
        program = PoolProcess(root, processalloc=processalloc, outdir=str(tmpdir))
        program._comm = comm
        for i in xrange(1000):
            jobs = program._getjobs()
            assert sum(program._alloc[u] for u in jobs) <= program._comm['n'],\
                (jobs, [program._alloc[u] for u in jobs])
