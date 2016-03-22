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
from pytest import fixture, mark
from pylada.espresso import Pwscf


@fixture
def aluminum(tmpdir):
    """ Creates input for aluminum """
    tmpdir.join('al.scf').write("""
        &control
           prefix='al'
           outdir='temporary directory for large files'
           pseudo_dir = '%s',
        /
        &system
           ibrav=  2, celldm(1) =7.50, nat=  1, ntyp= 1,
           ecutwfc =12.0,
           occupations='smearing', smearing='marzari-vanderbilt', degauss=0.06
        /
        &electrons
        /
       ATOMIC_SPECIES
        Al  26.98 Al.vbc.UPF
       ATOMIC_POSITIONS
        Al 0.00 0.00 0.00 
       K_POINTS automatic
         6 6 6 1 1 1
    """ % tmpdir.join('pseudos'))
    return str(tmpdir.join('al.scf'))


@fixture
def espresso():
    return Pwscf()


def test_attributes_default(espresso):
    assert espresso.control.calculation == 'scf'
    assert espresso.control.title is None
    assert espresso.control.verbosity == 'low'
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
    assert not hasattr(espresso.system, 'tot_charge')
    assert 'tot_charge' not in espresso.system.namelist()
    espresso.system.tot_charge = 1
    assert 'tot_charge' in espresso.system.namelist()


def test_read_aluminum(tmpdir, aluminum):
    from pylada.espresso import read_structure
    espresso = Pwscf()
    espresso.read(aluminum)
    structure = read_structure(aluminum)
    check_aluminum_functional(tmpdir, espresso)
    check_aluminum_structure(structure)


def check_aluminum_functional(tmpdir, espresso):
    from quantities import atomic_mass_unit
    from numpy import allclose
    assert espresso.control.prefix == 'al'
    assert espresso.control.outdir == 'temporary directory for large files'
    assert espresso.control.pseudo_dir == str(tmpdir.join('pseudos'))

    # atomic_species is a a private card, handled entirely by the functional 
    assert not hasattr(espresso, 'atomic_species')
    assert len(espresso.species) == 1
    assert 'Al' in espresso.species
    assert espresso.species['Al'].pseudo == 'Al.vbc.UPF'
    assert abs(espresso.species['Al'].mass - 26.98 * atomic_mass_unit) < 1e-8

    assert hasattr(espresso, 'k_points')
    assert espresso.kpoints.subtitle == 'automatic'
    assert espresso.kpoints.value == '6 6 6 1 1 1'


def check_aluminum_structure(structure):
    from quantities import bohr_radius
    from numpy import allclose, array
    assert len(structure) == 1
    assert structure[0].type == 'Al'
    assert allclose(structure[0].pos, [0e0, 0, 0])
    cell = 0.5 * array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]], dtype='float64').transpose()
    assert allclose(structure.cell, cell)
    assert abs(structure.scale - 7.5 * bohr_radius) < 1e-8


def test_read_write_loop(aluminum, tmpdir, espresso):
    from pylada.espresso import read_structure
    espresso.read(aluminum)
    espresso.control.pseudo_dir = str(tmpdir.join('pseudos'))
    tmpdir.join('pseudos', 'Al.vbc.UPF').ensure(file=True)
    structure = read_structure(aluminum)
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
