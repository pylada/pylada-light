###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to
#  submit large numbers of jobs on supercomputers. It provides a python interface to physical input,
#  such as crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential
#  programs. It is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU
#  General Public License as published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################
from pytest import fixture


@fixture
def root():
    from pylada.jobfolder import JobFolder
    from dummy import functional
    sizes = [10, 15, 20, 25]
    root = JobFolder()
    for type, trial, size in [('this', 0, 10), ('this', 1, 15), ('that', 2, 20), ('that', 1, 20)]:
        jobfolder = root / type / trial
        jobfolder.functional = functional
        jobfolder.params['indiv'] = size
        if type == 'that':
            jobfolder.params['value'] = True
    return root


@fixture
def expected_compute():
    return {'that/1': 20, 'that/2': 20, 'this/1': 15, 'this/2': 15, 'this/0': 10}


def test_subfolder_creation():
    from pylada.jobfolder import JobFolder
    root = JobFolder()
    jobfolder = root / 'this' / '0'
    assert jobfolder.name == "/this/0/"


def test_subfolder_creation_nonstring():
    from pylada.jobfolder import JobFolder
    root = JobFolder()
    jobfolder = root / 'this' / 0
    assert jobfolder.name == "/this/0/"


def test_contains(root):
    assert 'this/0' in root
    assert 'this/1' in root
    assert 'that/2' in root
    assert 'that/1' in root
    assert 'other' not in root


def test_getitem_and_contains(root):
    assert '0' in root['this']
    assert '1' in root['this']
    assert '1' in root['that']
    assert '2' in root['that']


def test_values_iteration(root):
    from dummy import functional
    for jobfolder in root.values():
        assert repr(jobfolder.functional) == repr(functional)


def test_getattr(root):
    assert getattr(root['this/0'], 'indiv', 0) == 10
    assert getattr(root['this/1'], 'indiv', 0) == 15
    assert getattr(root['that/1'], 'indiv', 0) == 20
    assert getattr(root['that/2'], 'indiv', 0) == 20
    assert not hasattr(root['this/0'], 'value')
    assert not hasattr(root['this/1'], 'value')
    assert getattr(root['that/1'], 'value', False)
    assert getattr(root['that/2'], 'value', False)


def test_key_iteration(root):
    for key, test in zip(root, ['that/1', 'that/2', 'this/0', 'this/1']):
        assert key == test

    for key, test in zip(root['that/1'].root, ['that/1', 'that/2', 'this/0', 'this/1']):
        assert key == test

    for key, test in zip(root['that'], ['1', '2']):
        assert key == test

    for key, test in zip(root['this'], ['0', '1']):
        assert key == test


def test_delete_subfolder(root):
    del root['that/2']
    assert 'that/2' not in root


def test_compute(tmpdir, root, expected_compute):
    for name, jobfolder in root.items():
        result = jobfolder.compute(outdir=str(tmpdir.join(name)))
        assert result.success
        assert tmpdir.join(name, 'OUTCAR').check()
        assert result.indiv == expected_compute[name]


def test_pickling_then_compute(tmpdir, root, expected_compute):
    from pickle import loads, dumps
    for name, jobfolder in root.items():
        result = jobfolder.compute(outdir=str(tmpdir.join(name)))
        assert result.success
        assert tmpdir.join(name, 'OUTCAR').check()
        assert result.indiv == expected_compute[name]


def test_deepcopy_then_compute(tmpdir, root, expected_compute):
    from copy import deepcopy
    for name, jobfolder in deepcopy(root).items():
        result = jobfolder.compute(outdir=str(tmpdir.join(name)))
        assert result.success
        assert tmpdir.join(name, 'OUTCAR').check()
        assert result.indiv == expected_compute[name]


def test_deepcopy(root):
    from copy import deepcopy
    jobfolder = deepcopy(root)
    for subfolder in root.values():
         assert subfolder.name in jobfolder
