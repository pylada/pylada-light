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


def test_smith_normal_form():
    from numpy import dot, all
    from numpy.random import randint
    from numpy.linalg import det
    from pylada.crystal.cutilities import smith_normal_form

    for i in range(50):
        cell = randint(-5, 5, size=(3, 3))
        while abs(det(cell)) < 1e-2:
            cell = randint(-5, 5, size=(3, 3))
        s, l, r = smith_normal_form(cell)
        assert all(dot(dot(l, cell), r) == s)


def test_snf_require_non_singular_matrix():
    from pylada.crystal.cutilities import smith_normal_form
    from numpy import array
    from pytest import raises
    from pylada import error
    with raises(error.ValueError):
        smith_normal_form(array([[0, 0, 0], [1, 2, 0], [3, 4, 5]]))


def is_integer(o, tolerance=1e-8):
    from numpy import allclose, floor
    return allclose(o, floor(o + 0.1), tolerance)


def test_gruber():
    from numpy import dot
    from numpy.linalg import inv
    from pylada.crystal.cutilities import gruber

    cell = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    lim = 5

    for a00 in [-1, 1]:
        for a10 in range(-lim, lim + 1):
            for a11 in [-1, 1]:
                for a20 in range(-lim, lim + 1):
                    for a21 in range(-lim, lim + 1):
                        for a22 in [-1, 1]:
                            a = [[a00, 0, 0], [a10, a11, 0], [a20, a21, a22]]
                            g = gruber(dot(cell, a))
                            assert is_integer(dot(inv(cell), g))
                            assert is_integer(dot(inv(g), cell))


def test_gruber_fails_on_singular_matrix():
    from numpy import array
    from pylada.crystal.cutilities import gruber
    from pylada import error
    from pytest import raises
    with raises(error.ValueError):
        gruber(array([[0, 0, 0], [1, 2, 0], [4, 5, 6]]))

def test_gruber_reached_maximum_iterations():
    from numpy import array
    from pylada.crystal.cutilities import gruber
    from pylada import error
    from pytest import raises

    with raises(error.RuntimeError):
        gruber(array([[1, 0, 0], [1, 1, 0], [4, 5, 1]]), itermax=2)
