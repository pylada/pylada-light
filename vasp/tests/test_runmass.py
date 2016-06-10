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
from pytest import fixture, mark
from pylada import vasp_program


@fixture
def path():
    from os.path import dirname
    return dirname(__file__)


@mark.skipif(vasp_program is None, reason="vasp not configured")
def test(tmpdir, path):
    from numpy import abs
    from pylada.crystal import Structure
    from pylada.vasp import Vasp
    from pylada.vasp.emass import effective_mass, EMass
    from pylada import default_comm

    structure = Structure([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], scale=5.55, name='has a name')\
        .add_atom(0, 0, 0, "Si")\
        .add_atom(0.25, 0.25, 0.25, "Si")

    vasp = Vasp()
    vasp.kpoints = "Automatic generation\n0\nMonkhorst\n2 2 2\n0 0 0"
    vasp.prec = "accurate"
    vasp.ediff = 25e-5
    vasp.encut = 1.4
    vasp.ismear = "fermi"
    vasp.sigma = 0.01
    vasp.relaxation = "volume"
    vasp.add_specie = "Si", "{0}/pseudos/Si".format(path)
    emass = EMass(copy=vasp)
    assert abs(emass.encut - 1.4) < 1e-8
    assert abs(emass.ediff - 25e-5) < 1e-10
    result = effective_mass(vasp, structure, outdir=str(tmpdir), comm=default_comm,
                            emassparams={'ediff': 1e-8})
    result.emass
    assert result.success
    result = emass(structure, outdir=str(tmpdir), comm=default_comm,
                   emassparams={'ediff': 1e-8})
    assert result.success
