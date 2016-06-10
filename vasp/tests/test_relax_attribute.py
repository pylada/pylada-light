###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is vasp high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides vasp python interface to physical input, such as
#  crystal structures, as well as to vasp number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR vasp PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received vasp copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################
from pytest import fixture, mark


@fixture
def vasp():
    from pylada.vasp import Vasp
    vasp = Vasp()
    for key in list(vasp._input.keys()):
        if key not in ['isif', 'nsw', 'ibrion', 'relaxation']:
            del vasp._input[key]
    return vasp


def test_fixture(vasp):
    """ Makes sure fixture is empty of anything but relaxation stuff """
    assert len(vasp.output_map(vasp=vasp)) == 1


def test_default_is_static(vasp):
    assert vasp.relaxation == 'static'
    assert vasp.ibrion is None
    assert vasp.isif is None
    assert vasp.nsw is None
    assert vasp.output_map(vasp=vasp)['ibrion'] == str(-1)


@mark.parametrize('prior', ['static', 'cellshape volume'])
def test_static(vasp, prior):
    vasp.relaxation = prior
    vasp.relaxation = 'static'
    assert vasp.relaxation == 'static'
    assert vasp.ibrion == -1 or vasp.ibrion is None
    assert vasp.nsw == 0 or vasp.nsw is None
    assert vasp.isif == 2 or vasp.isif is None
    assert vasp.output_map(vasp=vasp)['ibrion'] == str(-1)


def check_cellshape(vasp, nsw=50):
    assert vasp.relaxation == 'cellshape'
    assert vasp.isif == 5
    assert vasp.nsw == nsw
    assert vasp.ibrion == 2
    assert len(vasp.output_map(vasp=vasp)) == 3
    assert vasp.output_map(vasp=vasp)['ibrion'] == str(2)
    assert vasp.output_map(vasp=vasp)['isif'] == str(5)
    assert vasp.output_map(vasp=vasp)['nsw'] == str(nsw)


def test_cellshape(vasp):
    vasp.relaxation = 'cellshape'
    check_cellshape(vasp)
    vasp.nsw = 25
    check_cellshape(vasp, 25)


def test_cellshape_and_pickle(vasp):
    from pickle import loads, dumps
    vasp.relaxation = 'cellshape'
    check_cellshape(loads(dumps(vasp)))
    vasp.nsw = 25
    check_cellshape(loads(dumps(vasp)), 25)


def test_cellshape_volume(vasp):
    vasp.relaxation = 'cellshape volume'
    vasp.nsw = 25
    assert vasp.isif == 6
    assert vasp.relaxation == 'cellshape volume'
    assert vasp.nsw == 25
    assert vasp.ibrion == 2
    assert len(vasp.output_map(vasp=vasp)) == 3
    assert vasp.output_map(vasp=vasp)['ibrion'] == str(2)
    assert vasp.output_map(vasp=vasp)['isif'] == str(6)
    assert vasp.output_map(vasp=vasp)['nsw'] == str(25)


@mark.parametrize('relaxation', ['ions', 'ionic'])
def test_ions(vasp, relaxation):
    vasp.relaxation = relaxation
    assert vasp.relaxation == 'ionic'
    assert vasp.isif == 2


def test_cellshape_volume_ions(vasp):
    vasp.relaxation = 'cellshape, volume ions'
    assert vasp.relaxation == 'cellshape ionic volume'
    assert vasp.isif == 3


def test_cellshape_ionic(vasp):
    vasp.relaxation = 'cellshape, ionic'
    assert vasp.relaxation == 'cellshape ionic'
    assert vasp.isif == 4


def test_volume(vasp):
    vasp.relaxation = 'volume'
    assert vasp.relaxation == 'volume'
    assert vasp.isif == 7


def test_ions_volume_fails(vasp):
    from pytest import raises
    with raises(ValueError):
        vasp.relaxation = "ions, volume"
