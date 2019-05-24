###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab

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


def has_mpi4py():
    try:
        import mpi4py
        return True
    except ImportError:
        return False


mpi4py_required = mark.skipif(
    not has_mpi4py(), reason="mpi4py is not installed.")


@fixture
def executable():
    from os.path import join, dirname
    from pylada.process.tests.pifunctional import __file__ as executable
    return join(dirname(executable), "pifunctional.py")


@fixture
def comm():
    from pylada import default_comm
    result = default_comm.copy()
    result['n'] = 4
    return result


def jobfolders(executable, start=0, end=8):
    from pylada.process.tests.functional import Functional
    from pylada.jobfolder.jobfolder import JobFolder
    root = JobFolder()
    for n in range(start, end):
        job = root / str(n)
        job.functional = Functional(executable, [n])
        job.params['sleep'] = 0.01
    return root
