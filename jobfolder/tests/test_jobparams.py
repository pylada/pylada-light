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
        jobfolder = root / type / str(trial)
        jobfolder.functional = functional
        jobfolder.params['indiv'] = size
        if type == 'that':
            jobfolder.params['value'] = True

    return root


@fixture
def jobparams(jobfolder):
    from pylada.jobfolder import JobParams
    return JobParams(jobfolder)


@fixture
def extra_folder(functional):
    from pylada.jobfolder import JobFolder
    jobfolder = JobFolder()
    jobfolder.functional = functional
    jobfolder.params['indiv'] = 25
    jobfolder.params['value'] = 5
    jobfolder.params['another'] = 6
    return jobfolder


def test_attribute_forwarding(jobparams):
    assert len(jobparams.functional.values()) == 4


def test_enumeration_of_functional(functional, jobparams):
    for i, (name, value) in enumerate(jobparams.functional.items()):
        assert repr(value) == repr(functional)
    assert i == 3


def test_enumeration_of_parameter_that_all_jobs_have(jobparams):
    expected = {'/that/1/': 20, '/that/2/': 20, '/this/1/': 15, '/this/2/': 15, '/this/0/': 10}
    for i, (name, value) in enumerate(jobparams.indiv.items()):
        assert value == expected[name]
    assert i == 3


def test_enumeration_of_parameter_that_only_a_few_jobs_have(jobparams):
    expected = {'/that/2/': True, '/that/1/': True}
    for i, (name, value) in enumerate(jobparams.value.items()):
        assert value == expected[name]
    assert i == 1


def test_accessing_unknown_attribute_fails(jobparams):
    from pytest import raises

    with raises(AttributeError):
        jobparams.that


@mark.parametrize('regex, keys', [
    ('*/1', ('/this/1/', '/that/1/')),
    ('this', ('/this/1/', '/this/0/')),
    ('that/2/', ('/that/2/',))
])
def test_wildcard_indexing(regex, keys, jobparams):
    jobparams.unix_re = True
    assert set((u for u in jobparams[regex].keys())) == set(keys)


def test_add_item(extra_folder, jobparams, functional):
    jobparams['/this/0/another'] = extra_folder
    assert '/this/0/another' in jobparams
    assert getattr(jobparams['/this/0/another'], 'indiv', 0) == 25
    assert getattr(jobparams['/this/0/another'], 'value', 0) == 5
    assert getattr(jobparams['this/0/another'], 'another', 0) == 6
    assert isinstance(jobparams['this/0/another'].functional, type(functional))


@mark.parametrize('regex, keys, path', [
    ('*/*/another', ['/this/0/another/'], '/'),
    ('*/*/another/', ['/this/0/another/'], '/'),
    ('*/another', [], '/'),
    ('../that', ['/that/2/', '/that/1/'], 'this'),
    ('../that', [], 'this/0'),
    ('../*', ['/this/0/', '/this/1/', '/this/0/another/'], 'this/0'),
    ('../*', ['/this/0/', '/this/1/', '/this/0/another/'], 'this/*')
])
def test_more_complex_wildcard_indexing(regex, keys, jobparams, extra_folder, path):
    jobparams['/this/0/another'] = extra_folder
    jobparams.unix_re = True
    subfolders = jobparams[path]
    assert set((u for u in subfolders[regex].keys())) == set(keys)


@mark.parametrize('regex, keys', [
    ('.*/1', ['/this/1/', '/that/1/']),
    ('this', ['/this/1/', '/this/0/', '/this/0/another/']),
    ('that/2/', ['/that/2/']),
    ('.*/.*/another', ['/this/0/another/']),
    ('.*/another', ['/this/0/another/'])
])
def test_regex_indexing(jobparams, regex, keys, extra_folder):
    jobparams['/this/0/another'] = extra_folder
    jobparams.unix_re = False
    assert set((u for u in jobparams[regex].keys())) == set(keys)


def test_naked_end(jobparams, extra_folder):
    jobparams['/this/0/another'] = extra_folder
    jobparams.unix_re = True
    jobparams.naked_end = True
    assert isinstance(jobparams.another, int)


def test_attribute_enumeration_with_wildcard(jobparams):
    jobparams.unix_re = True
    expected = {'/this/1/': 15, '/that/1/': 20}
    for i, (key, value) in enumerate(jobparams['*/1'].indiv.items()):
        assert value == expected[key]
    assert i == 1


def test_setting_attribute_for_all_jobs(jobparams):
    jobparams.indiv = 5
    for i, (key, value) in enumerate(jobparams.indiv.items()):
        assert value == 5

    assert i == 3


def test_setting_attribute_for_some_jobs(jobparams, extra_folder):
    jobparams['/this/0/another'] = extra_folder
    jobparams['*/0'].indiv = 6
    expected = {
        '/this/0/': 6, '/this/1/': 15, '/that/2/': 20, '/that/1/': 20, '/this/0/another/': 6
    }
    for i, (key, value) in enumerate(jobparams.indiv.items()):
        assert value == expected[key]

    assert i == 4


def test_deleting_folder(jobparams):
    del jobparams['/*/1']
    assert '/this/1/' not in jobparams
    assert '/that/1/' not in jobparams
    assert '/that/2/' in jobparams
    assert '/this/0/' in jobparams


def test_concatenate_jobfolders(jobparams, jobfolder, functional, extra_folder):
    from copy import deepcopy
    from pylada.jobfolder import JobFolder, JobParams

    # jobparams owns a reference to jobfolder
    # make this a different object before adding extra_folder
    # so we can check concatenate does not affect subfolders not in the concatenated jobfolder
    jobfolder = deepcopy(jobfolder)
    jobparams['/this/0/another'] = extra_folder

    # change values of all individuals
    jobparams.indiv = 2
    # this should reset them except for the extra folder (since not in jobfolder)
    jobparams.concatenate(jobfolder)

    for name, value in jobparams.functional.items():
        assert repr(value) == repr(functional)

    expected = {
        '/this/0/': 10, '/this/1/': 15, '/that/1/': 20, '/that/2/': 20, '/this/0/another/': 2
    }
    for key, value in jobparams.indiv.items():
        assert value == expected[key]


def test_concatenate_jobparams_and_indexing(jobparams, jobfolder, functional, extra_folder):
    from copy import deepcopy
    from pylada.jobfolder import JobFolder, JobParams

    # jobparams owns a reference to jobfolder
    # make this a different object before adding extra_folder
    # so we can check concatenate does not affect subfolders not in the concatenated jobfolder
    jobfolder = deepcopy(jobfolder)
    jobparams['/this/0/another'] = extra_folder

    del jobparams['/*/1']
    jobparams.indiv = 2
    jobparams.concatenate(JobParams(jobfolder)['/*/1'])

    for name, value in jobparams.functional.items():
        assert repr(value) == repr(functional)

    expected = {
        '/this/0/': 2, '/this/1/': 15, '/that/1/': 20, '/that/2/': 2, '/this/0/another/': 2
    }
    for key, value in jobparams.indiv.items():
        assert value == expected[key]
