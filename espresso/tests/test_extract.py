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

# -*- coding: utf-8 -*-
""" Pwscf Extraction Tests """
from py.test import fixture, mark
from pylada.espresso.extract import Extract
import pylada.espresso.tests.fixtures
aluminum_pwscf = pylada.espresso.tests.fixtures.aluminum_pwscf
aluminum_structure = pylada.espresso.tests.fixtures.aluminum_structure


@fixture
def ions_path():
    from py.path import local
    return local(local(__file__).dirname).join("data", "ions")


@fixture
def ions(ions_path):
    return Extract(ions_path)


@fixture
def nonscf_path():
    from py.path import local
    return local(local(__file__).dirname).join("data", "nonscf")


@fixture
def nonscf(nonscf_path):
    return Extract(nonscf_path)


def test_is_running(tmpdir):
    extract = Extract(tmpdir)
    assert not extract.is_running
    tmpdir.ensure(".pylada_is_running")
    assert extract.is_running


def test_abspath(tmpdir):
    from os import environ
    environ['FAKE_ME_OUT'] = str(tmpdir)
    extract = Extract("$FAKE_ME_OUT/thishere")
    assert extract.abspath == str(tmpdir.join("thishere"))


@mark.parametrize("prefix", ['pwscf', 'biz'])
def test_success(tmpdir, prefix):
    extract = Extract(tmpdir, prefix)

    if prefix is None:
        prefix = 'pwscf'

    # file does not exist
    assert extract.success == False

    # file does exist but does not contain JOB DONE
    path = tmpdir.join("%s.out" % prefix)
    path.ensure(file=True)
    assert extract.success == False

    path.write("HELLO")
    assert extract.success == False

    path.write("JOB DONE.")
    assert extract.success


def test_functional(nonscf, aluminum_pwscf):
    from numpy import abs
    expected = aluminum_pwscf
    pwscf = nonscf.functional
    assert abs(pwscf.system.ecutwfc - expected.system.ecutwfc) < 1e-8
    assert pwscf.system.occupations == expected.system.occupations


@mark.parametrize('attr', ['structure', 'initial_structure'])
def test_initial_structure(nonscf, aluminum_structure, attr):
    from numpy import allclose, abs
    structure = getattr(nonscf, attr)
    assert len(structure) == len(aluminum_structure)
    assert allclose(structure.cell, aluminum_structure.cell)
    assert abs(structure.scale - aluminum_structure.scale) < 1e-8
    for atom, expected in zip(structure, aluminum_structure):
        assert allclose(atom.pos, expected.pos)
        assert atom.type == expected.type


def test_ions_structure(ions):
    from numpy import allclose, abs
    structure = ions.structure
    print(structure)
    assert allclose(structure[0].pos, [-0.002382255, -0.002360252, 0.002263443], 1e-5)
    assert allclose(structure[1].pos, [0.247615432, 0.247636961, 0.252268307], 1e-5)
    assert allclose(structure[0].force.magnitude, [-0.00000632, -0.00000775, 0.00001397])
    assert allclose(structure[1].force.magnitude, [0.00000632, 0.00000775, -0.00001397])
    assert allclose(structure.cell, ions.initial_structure.cell, 1e-12)
    assert abs(structure.scale - ions.initial_structure.scale) < 1e-12


def test_forces(ions):
    from numpy import allclose
    from quantities import Ry, bohr_radius as a0
    assert ions.forces.shape[0] == 2
    assert ions.forces.shape[1] == 3
    assert ions.forces.simplified.dimensionality == (Ry / a0).simplified.dimensionality
    assert ions.forces.units == (Ry / a0).units
    assert allclose(ions.forces.magnitude[0], [-0.00000632, -0.00000775, 0.00001397])
    assert allclose(ions.forces.magnitude[1], [0.00000632, 0.00000775, -0.00001397])
