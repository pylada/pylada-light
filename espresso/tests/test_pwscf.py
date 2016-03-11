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
           pseudo_dir = 'directory where pp-files are kept',
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
    """)
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


def test_read_aluminum(aluminum):
    espresso = Pwscf()
    espresso.read(aluminum)
    check_aluminum(espresso)


def check_aluminum(espresso):
    from numpy import allclose
    assert espresso.control.prefix == 'al'
    assert espresso.control.outdir == 'temporary directory for large files'
    assert espresso.control.pseudo_dir == 'directory where pp-files are kept'
    assert espresso.system.ibrav == 2
    assert allclose(espresso.system.celldm, [7.5])
    assert espresso.system.nat == 1
    assert espresso.system.ntyp == 1

    assert hasattr(espresso, 'atomic_species')
    assert espresso.atomic_species.subtitle is None
    assert espresso.atomic_species.value == 'Al  26.98 Al.vbc.UPF'

    assert hasattr(espresso, 'atomic_positions')
    assert espresso.atomic_positions.subtitle is None
    assert espresso.atomic_positions.value == 'Al 0.00 0.00 0.00'

    assert hasattr(espresso, 'k_points')
    assert espresso.kpoints.subtitle == 'automatic'
    assert espresso.kpoints.value == '6 6 6 1 1 1'


def test_read_write_loop(aluminum, tmpdir):
    from pylada.espresso import logger
    logger.setLevel(31)
    espresso = Pwscf()
    espresso.read(aluminum)
    espresso.write(str(tmpdir.join('al2.scf')))
    espresso = Pwscf()
    espresso.read(str(tmpdir.join('al2.scf')))
    check_aluminum(espresso)

