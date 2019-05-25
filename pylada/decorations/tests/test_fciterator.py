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


def test_fciterator():
    from pylada.decorations._decorations import FCIterator
    result = [False, False, False, True, True], \
             [False, False, True, False, True], \
             [False, True, False, False, True], \
             [True, False, False, False, True], \
             [False, False, True, True, False], \
             [False, True, False, True, False], \
             [True, False, False, True, False], \
             [False, True, True, False, False], \
             [True, False, True, False, False], \
             [True, True, False, False, False]
    iterator = FCIterator(5, 2)
    for i, u in enumerate(iterator):
        assert all(u == result[i])
    iterator.reset()
    reit = False
    for i, u in enumerate(iterator):
        assert all(u == result[i])
        reit = True
    assert reit
