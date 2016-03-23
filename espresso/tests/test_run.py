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
def pwscf():
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
def aluminum():
    from quantities import bohr_radius
    from pylada.crystal.bravais import fcc
    result = fcc()
    result.scale = 7.5 * bohr_radius
    result[0].type = 'Al'
    return result

