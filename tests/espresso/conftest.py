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


@fixture
def aluminum_file(tmpdir):
    """ Creates input for aluminum """
    tmpdir.join('al.scf').write("""
        &control
           prefix='al',
           outdir='%s',
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
    """ % (tmpdir, tmpdir.join('pseudos')))
    return str(tmpdir.join('al.scf'))


def check_aluminum_functional(tmpdir, espresso):
    from quantities import atomic_mass_unit
    assert espresso.control.prefix == 'al'
    assert espresso.control.outdir == str(tmpdir)
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


@fixture
def aluminum_pwscf():
    from pylada.espresso import Pwscf
    pwscf = Pwscf()
    pwscf.system.ecutwfc = 12.0
    pwscf.system.occupations = 'smearing'
    pwscf.system.smearing = 'marzari-vanderbilt'
    pwscf.system.degauss = 0.06
    pwscf.kpoints.subtitle = 'automatic'
    pwscf.kpoints.value = '6 6 6 1 1 1'
    pwscf.add_specie('Al', 'Al.pz-vbc.UPF')
    return pwscf


@fixture
def aluminum_structure():
    from quantities import bohr_radius
    from pylada.crystal.bravais import fcc
    result = fcc()
    result.scale = 7.5 * bohr_radius
    result[0].type = 'Al'
    return result


@fixture
def diamond_structure():
    from quantities import bohr_radius
    from pylada.crystal.bravais import fcc
    result = fcc()
    result.scale = 10.2 * bohr_radius
    result[0].type = 'Si'
    result.add_atom(0.25, 0.25, 0.25, 'Si')
    return result


@fixture
def diamond_pwscf():
    from pylada.espresso import Pwscf
    pwscf = Pwscf()
    pwscf.system.ecutwfc = 12.0
    pwscf.kpoints.subtitle = None
    pwscf.kpoints.value = "2\n"\
        "0.25 0.25 0.75 3.0\n"\
        "0.25 0.25 0.25 1.0\n"
    pwscf.add_specie('Si', 'Si.pz-vbc.UPF')
    return pwscf


def run_nonscf(directory="nonscf"):
    """ Example of running non-scf calculations """
    structure = aluminum_structure()
    pwscf = aluminum_pwscf()
    pwscf.control.calculation = 'relax'
    pwscf(structure, outdir=directory)


def run_ions(directory="ions"):
    """ Example of running ionic relaxation calculations """
    from numpy.random import random
    structure = diamond_structure()
    pwscf = diamond_pwscf()

    structure[1].pos += random(3) * 0.01 - 0.005
    pwscf.control.calculation = 'relax'
    pwscf(structure, outdir=directory)


def run_cellrelax(directory="cellshape"):
    """ Example of running relaxation with cell-shape """
    from numpy.random import random
    structure = diamond_structure()
    pwscf = diamond_pwscf()

    structure[1].pos += random(3) * 0.01 - 0.005
    structure.cell += random((3, 3)) * 0.01 - 0.005
    pwscf.control.calculation = 'vc-relax'
    pwscf.cell.factor = 2.0
    pwscf(structure, outdir=directory)


def run_with_restart(outdir="restarting"):
    """ Example of running relaxation with cell-shape """
    from pylada.misc import local_path
    from numpy.random import random
    outdir = local_path(outdir)
    structure = diamond_structure()
    pwscf = diamond_pwscf()

    structure[1].pos += random(3) * 0.01 - 0.005
    structure.cell += random((3, 3)) * 0.01 - 0.005
    pwscf.control.calculation = 'vc-relax'
    pwscf.cell.factor = 2.0
    result = pwscf(structure, outdir=outdir.join('first'))
    result = pwscf(structure, outdir=outdir.join('second'), restart=result)
