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


def test_makeclass():
    """ Creating a functional from a function """
    from pylada.tools.makeclass import makeclass

    global iterator
    base = Base(1, 'b')

    Functional = makeclass('Functional', Base, iter_func, call=func)
    functional = Functional(b=-5, other=False)
    assert getattr(functional, 'other', True) == False
    assert functional(True) == False
    assert functional.a == 2
    assert iterator == 4
    assert functional.b == -5

    functional = Functional(b=-5, other=False, copy=base)
    assert functional.a == 1
    assert getattr(functional, 'other', True) == False
    assert functional(True) == False
    assert functional.a == 1
    assert iterator == 4
    assert functional.b == -5

    Functional = makeclass('Functional', Base, iter_func)
    functional = Functional(b=-5, other=False)
    assert getattr(functional, 'other', True) == False
    assert functional(True) is None
    assert functional.a == 2
    assert iterator == 4
    assert functional.b == -5

    functional = Functional(b=-5, other=False, copy=base)
    assert functional.a == 1
    assert getattr(functional, 'other', True) == False
    assert functional(True) is None
    assert functional.a == 1
    assert iterator == 4
    assert functional.b == -5


class Base(object):

    def __init__(self, a=2, b='b'):
        super(Base, self).__init__()
        self.a = a
        self.b = b

    def __call__(self, structure, outdir=None):
        self.b += 1

iterator = -1


class DummyProcess:

    def __init__(self, value): self.value = value

    def wait(self): return True

    def start(self, comm): return False


def iter_func(self, structure, outdir=None, other=True):
    assert other == False
    global iterator
    for iterator in range(1, 5):
        self.a *= iterator
        self(structure, outdir)
        yield DummyProcess(self)


def func(self, structure, outdir=None, comm=None, other=True):
    assert other == False
    self(structure, outdir)
    a = self.a
    for i, f in enumerate(iter_func(self, structure, outdir, other=other)):
        assert f.value.a / a == i + 1
        a = f.value.a
    self.a = 0
    return not structure
