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
from pytest import fixture, mark


@fixture
def functional():
    from dummy import functional
    return functional


@fixture
def jobfolder(functional):
    from pylada.jobfolder import JobFolder
    root = JobFolder()
    for type, trial, size in [('this', 0, 10), ('this', 1, 15), ('that', 2, 20), ('that', 1, 20)]:
        job = root / type / str(trial)
        job.functional = functional
        job.params['indiv'] = size
        if type == 'that':
            job.params['value'] = True

    job = root / 'this' / '0' / 'another'
    job.functional = functional
    job.params['indiv'] = 25
    job.params['value'] = 5
    job.params['another'] = 6
    return root


@fixture
def expected_results():
    return {
        '/this/0/': 10, '/this/1/': 15, '/that/1/': 20, '/that/2/': 20, '/this/0/another/': 25
    }


@fixture
def collect(tmpdir, jobfolder, expected_results):
    from pickle import dump
    from pylada.jobfolder import MassExtract
    # compute results
    for name, job in jobfolder.items():
        result = job.compute(outdir=str(tmpdir.join(name)))
        assert result.success
        assert result.indiv == expected_results['/' + name + '/']

    # and pickle the jobfolder
    with open(str(tmpdir.join('dict')), 'w') as file:
        dump(jobfolder, file)

    # now ready to extract stuff
    return MassExtract(path=str(tmpdir.join('dict')))


def test_functional_is_collected(functional, collect):
    for i, (name, value) in enumerate(collect.functional.items()):
        assert repr(value) == repr(functional)

    assert i == 4


def test_indiv_is_collected(collect, expected_results):
    for name, value in collect.indiv.items():
        assert value == expected_results[name]


def test_fail_on_unknown_attribute(collect):
    from pytest import raises
    with raises(AttributeError):
        collect.that


@mark.parametrize('regex, keys', [
    ('*/1', ['/this/1/', '/that/1/']),
    ('this', ['/this/1/', '/this/0/', '/this/0/another/']),
    ('that/2/', ['/that/2/'])
])
def test_wildcard_indexing(keys, regex, collect):
    assert set(collect[regex].keys()) == set(keys)


@mark.parametrize('regex, keys, path', [
    ('*/*/another', ['/this/0/another/'], '/'),
    ('*/*/another/', ['/this/0/another/'], '/'),
    ('*/another', [], '/'),
    ('../that', ['/that/2/', '/that/1/'], 'this'),
    ('../that', [], 'this/0'),
    ('../*', ['/this/0/', '/this/1/', '/this/0/another/'], 'this/0'),
    ('../*', ['/this/0/', '/this/1/', '/this/0/another/'], 'this/*')
])
def test_more_complex_wildcard_indexing(keys, regex, path, collect, jobfolder):
    # add empty item to jobfolder
    job = jobfolder / 'this' / '0' / 'another'
    # get subfolders
    subfolders = collect[path]
    assert set(subfolders[regex].keys()) == set(keys)


@mark.parametrize('regex, keys', [
    ('.*/1', ['/this/1/', '/that/1/']),
    ('this', ['/this/1/', '/this/0/', '/this/0/another/']),
    ('that/2/', ['/that/2/']),
    ('.*/.*/another', ['/this/0/another/']),
    ('.*/another', ['/this/0/another/'])
])
def test_regex_indexing(keys, regex, collect, jobfolder):
    collect.unix_re = False
    # add empty item to jobfolder
    job = jobfolder / 'this' / '0' / 'another'
    assert set(collect[regex].keys()) == set(keys)


def test_naked_end(jobfolder, functional, tmpdir, expected_results, collect):
    job = jobfolder / 'this' / '0' / 'another'
    collect.naked_end = False
    for key, value in collect['*/1'].indiv.items():
        assert value == expected_results[key]
