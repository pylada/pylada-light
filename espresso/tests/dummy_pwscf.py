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
from pylada.espresso import Pwscf
from tempfile import NamedTemporaryFile
from pylada.espresso.tests.fixtures import check_aluminum_functional, check_aluminum_structure
from pylada.espresso import read_structure
from py.path import local
from sys import stdin
pwscf = Pwscf()
with NamedTemporaryFile(mode="w") as file:
    file.write(stdin.read())
    file.flush()
    pwscf.read(file.name)
    structure = read_structure(file.name)

check_aluminum_functional(local(), pwscf)
check_aluminum_structure(structure)
pwscf.pseudos_do_exist(structure)
print("JOB IS DONE!")
