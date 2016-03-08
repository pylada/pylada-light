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


def test_common():
    from os.path import join, dirname
    from numpy import array, all, abs
    from pylada.vasp import Extract
    import pylada
    from quantities import eV, angstrom

    a = Extract(directory=join(dirname(__file__), 'data', 'COMMON'))
    assert a.success == True
    assert a.algo == "Fast"
    assert a.is_dft == True
    assert a.is_gw == False
    assert abs(a.encut - 245.3 * eV) < 1e-8
    assert repr(a.datetime) == 'datetime.datetime(2012, 3, 8, 21, 18, 29)'
    assert a.LDAUType is None
    assert len(a.HubbardU_NLEP) == 0
    assert a.pseudopotential == 'PAW_PBE'
    assert set(a.stoichiometry) == set([2])
    assert set(a.species) == set(['Si'])
    assert a.isif == 7
    assert abs(a.sigma - 0.2 * eV) < 1e-8
    assert a.nsw == 50
    assert a.ibrion == 2
    assert a.relaxation == "volume"
    assert a.ispin == 1
    assert a.isym == 2
    assert a.name == 'has a name'
    assert a.system == "has a name"
    assert all(abs(array(a.ionic_charges) - [4.0]) < 1e-8)
    assert abs(a.nelect - 8.0) < 1e-8
    assert abs(a.extraelectron - 0e0) < 1e-8
    assert abs(a.nbands - 8) < 1e-8
    assert all(abs(array([[0, 2.73612395, 2.73612395], [2.73612395, 0, 2.73612395], [
               2.73612395, 2.73612395, 0]]) - a.structure.cell) < 1e-4)
    assert all(abs(a.structure.scale - 1.0 * angstrom) < 1e-4)
    assert all(abs(a.structure.scale * a.structure.cell - a._grep_structure.cell * angstrom) < 1e-4)
    assert all(abs(a.structure[0].pos) < 1e-8)
    assert all(abs(a.structure[1].pos - 1.36806) < 1e-6)
    assert all([b.type == 'Si' for b in a.structure])
    assert abs(a.structure.energy + 10.665642 * eV) < 1e-6
    assert abs(a.sigma - 0.2 * eV) < 1e-6
    assert abs(a.ismear - 1) < 1e-6
    assert abs(a.potim - 0.5) < 1e-6
    assert abs(a.istart - 0) < 1e-6
    assert abs(a.icharg - 2) < 1e-6
    assert a.precision == "accurate"
    assert abs(a.ediff - 2e-5) < 1e-8
    assert abs(a.ediffg - 2e-4) < 1e-8
    assert a.lorbit == 0
    assert abs(a.nupdown + 1.0) < 1e-8
    assert a.lmaxmix == 4
    assert abs(a.valence - 8.0) < 1e-8
    assert a.nonscf == False
    assert a.lwave == False
    assert a.lcharg
    assert a.lvtot == False
    assert a.nelm == 60
    assert a.nelmin == 2
    assert a.nelmdl == -5
    assert all(abs(a.kpoints - array([[0.25,  0.25,  0.25], [0.75, -0.25, -0.25]])) < 1e-8)
    assert all(abs(a.multiplicity - [96.0, 288.0]) < 1e-8)
    pylada.verbose_representation = True
