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
from pytest import fixture
from pylada.espresso import Pwscf
from pylada.espresso.tests.fixtures import check_aluminum_functional, check_aluminum_structure
import pylada.espresso.tests.fixtures
aluminum_file = pylada.espresso.tests.fixtures.aluminum_file


@fixture
def espresso():
    pwscf = Pwscf()
    # Required for all writing because required by Pwscf
    pwscf.system.ecutrho = 12.0
    return pwscf


def test_attributes_default(espresso):
    assert espresso.control.calculation is None
    assert espresso.control.title is None
    assert espresso.control.verbosity is None
    assert espresso.system.nbnd is None
    assert len(espresso.electrons) == 0
    assert espresso.kpoints.name == 'k_points'
    assert espresso.kpoints.subtitle == 'gamma'
    assert espresso.kpoints.value is None


def test_traits_do_fail(espresso):
    from traitlets import TraitError
    from pytest import raises
    with raises(TraitError):
        espresso.control.calculation = 'whatever'

    with raises(TraitError):
        espresso.system.nbnd = 1.3


def test_can_set_attributes(espresso):
    espresso.control.calculation = 'nscf'
    assert espresso.control.calculation == 'nscf'
    espresso.system.nbnd = 1
    assert espresso.system.nbnd == 1


def test_can_add_namelist_attributes(espresso):
    assert not hasattr(espresso.system, 'toot_charge')
    assert 'toot_charge' not in espresso.system.namelist()
    espresso.system.toot_charge = 1
    assert 'toot_charge' in espresso.system.namelist()


def test_read_aluminum(tmpdir, aluminum_file):
    from pylada.espresso import read_structure
    espresso = Pwscf()
    espresso.read(aluminum_file)
    structure = read_structure(aluminum_file)
    check_aluminum_functional(tmpdir, espresso)
    check_aluminum_structure(structure)


def test_read_write_loop(aluminum_file, tmpdir, espresso):
    from pylada.espresso import read_structure
    espresso.read(aluminum_file)
    espresso.control.pseudo_dir = str(tmpdir.join('pseudos'))
    tmpdir.join('pseudos', 'Al.vbc.UPF').ensure(file=True)
    structure = read_structure(aluminum_file)
    espresso.write(str(tmpdir.join('al2.scf')), structure=structure)
    espresso = Pwscf()

    espresso.read(str(tmpdir.join('al2.scf')))
    check_aluminum_functional(tmpdir, espresso)


def test_bringup(tmpdir, espresso):
    from pylada.crystal.A2BX4 import b5
    structure = b5()
    # create fake pseudo files: _bring_up checks that the files exist
    tmpdir.join('pseudos', 'A.upf').ensure(file=True)
    tmpdir.join('pseudos', 'B.upf').ensure(file=True)
    tmpdir.join('pseudos', 'X.upf').ensure(file=True)

    espresso.control.pseudo_dir = str(tmpdir.join('pseudos'))
    espresso.add_specie('A', 'A.upf', mass=1)
    espresso.add_specie('B', 'B.upf', mass=2)
    espresso.add_specie('X', 'X.upf', mass=3)
    espresso._bring_up(outdir=str(tmpdir.join('runhere')), structure=structure)
    assert tmpdir.join('runhere', 'pwscf.in').check()


def test_atomic_specie(tmpdir, espresso):
    from pylada.crystal.A2BX4 import b5
    structure = b5()

    tmpdir.join('pseudos', 'A.upf').ensure(file=True)
    tmpdir.join('pseudos', 'B.upf').ensure(file=True)
    tmpdir.join('pseudos', 'X.upf').ensure(file=True)

    espresso.control.pseudo_dir = str(tmpdir.join('pseudos'))
    espresso.add_specie('A', 'A.upf', mass=1)
    espresso.add_specie('B', 'B.upf', mass=2)
    espresso.add_specie('X', 'X.upf', mass=3)

    card = espresso._write_atomic_species_card(structure)
    assert card.name == "atomic_species"
    assert card.subtitle is None
    lines = card.value.rstrip().lstrip().split('\n')
    assert len(lines) == 3
    assert all(len(u.split()) == 3 for u in lines)
    assert set([u.split()[0] for u in lines]) == {'A', 'B', 'X'}
    assert set([int(float(u.split()[1])) for u in lines]) == {1, 2, 3}
    assert set([u.split()[2] for u in lines]) == {'A.upf', 'B.upf', 'X.upf'}


def test_iteration(tmpdir, aluminum_file, espresso):
    """ Checks iterations goes through the expected steps """
    from sys import executable as python
    from os.path import dirname, join
    from pylada.espresso import read_structure
    from pylada.espresso.tests.extract import Extract
    structure = read_structure(aluminum_file)
    espresso.read(aluminum_file)
    espresso.program = python + " " + join(dirname(__file__), 'dummy_pwscf.py')
    espresso.Extract = Extract
    tmpdir.join('pseudos', 'Al.vbc.UPF').ensure(file=True)
    iterator = espresso.iter(outdir=str(tmpdir), overwrite=True, structure=structure)
    program_process = next(iterator)
    assert hasattr(program_process, 'start')
    assert hasattr(program_process, 'wait')
    program_process.start()
    program_process.wait()
    assert tmpdir.join("%s.out" % espresso.control.prefix).check()
    extract = next(iterator)
    assert isinstance(extract, Extract)
    assert extract.success


def test_add_namelist(espresso):
    espresso.add_namelist("nml", wtd=2)
    assert hasattr(espresso, 'nml')
    assert getattr(espresso.nml, 'wtd', 3) == 2


def test_add_existing_namelist(espresso):
    espresso.electrons.cat = 2
    espresso.add_namelist("electrons", wtd=2)
    assert hasattr(espresso, 'electrons')
    assert not hasattr(espresso.electrons, 'cat')
    assert getattr(espresso.electrons, 'wtd', 3) == 2


def test_ecutrho_required():
    from py.test import raises
    from pylada.espresso import Pwscf
    from pylada import error
    pwscf = Pwscf()

    with raises(error.ValueError):
        pwscf.system.namelist()

    with raises(error.ValueError):
        pwscf.write()

def test_dimensional_trait_transform(espresso):
    from numpy import abs
    from quantities import eV, Ry
    espresso.system.ecutrho = 100 * eV

    assert abs(espresso.system.ecutrho - 100 * eV) < 1e-8
    assert espresso.system.ecutrho.units == eV
    assert abs(espresso.system.namelist()['ecutrho'] - float((100 * eV).rescale(Ry))) < 1e-8
