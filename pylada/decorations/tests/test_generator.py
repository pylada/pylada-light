###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
#  crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################
from pytest import mark
from itertools import chain
from pylada.decorations.tests import fccsets
from pylada.decorations.tests import ternarysets
from pylada.decorations.tests import diamondsets
from pylada.decorations.tests import zincblendesets


@mark.parametrize('lattice, natoms, expected', chain(
    ((fccsets.lattice, u, v) for u, v in fccsets.datasets.items()),
    ((ternarysets.lattice, u, v) for u, v in ternarysets.datasets.items()),
    ((diamondsets.lattice, u, v) for u, v in diamondsets.datasets.items()),
    ((zincblendesets.lattice, u, v) for u, v in zincblendesets.datasets.items()),
))
def test_generator(lattice, natoms, expected):
    from pylada.decorations import generate_bitstrings

    assert len(expected) > 0
    result = []
    for x, hft, hermite in generate_bitstrings(lattice, [natoms]):
        result.append(''.join(str(i) for i in hermite.flatten()[[0, 3, 4, 6, 7, 8]])
                      + ' ' + ''.join(str(i - 1) for i in x))

    assert len(result) == len(expected)
    assert set(result) == expected
