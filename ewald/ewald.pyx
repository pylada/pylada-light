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

""" Point-ion charge models.

    This sub-package provides an ewald summation routine and a lennard-johnes
    potential.  The implementations are fairly basic run-of-the-mill fortran
    stuff with a python interface.
"""

__docformat__ = "restructuredtext en"
__all__ = ['ewald']

cdef extern from "ewald/ewald.h" namespace "pylada":
    int ewaldc(int verbosity, double &energy, double * reduced_forces, double *cartesian_forces,
               double * stress, int natoms, double * reduced_atomic_coords, double * atomic_charges,
               double real_space_cutoff, double * cell_vectors);

def ewald(structure, charges=None, cutoff=15, verbose=False, **kwargs):
    """ Ewald summation.

        Run-of-the-mill Ewald summation. Nothing fancy, so not very fast for
        large structures.

        :param structure:
            The structure to optimize. The charge of each atom can be given as a ``charge`` attribute.
            Otherwise, they should be in the ``charges`` map.

        :type structure:
            py:attr:`~pylada.crystal.Structure`

        :param dict charges:
            Map from atomic-types to charges. If not signed by a unit, then should be in units of
            elementary electronic charge. If an atom has a ``charge`` attribute, the attribute takes
            priority ove items in this map.

        :param float cutoff:
            Cutoff energy when computing reciprocal space part. Defaults to :py:math:`15 Ry`.

    """
    from . import physics, error
    from numpy import array, dot, zeros, require
    from numpy.linalg import inv
    from quantities import elementary_charge as em, Ry, a0, angstrom

    cell = require(structure.cell.copy(), dtype='float64', requirements=['F_CONTIGUOUS']) 
    cell *= float(structure.scale.rescale(a0))

    if hasattr(cutoff, 'rescale'):
        cutoff = float(cutoff.rescale(Ry))

    if charges is None:
        charges = {}

    def get_charge(atom):
        from quantities import elementary_charge
        charge = getattr(atom, 'charge', charges.get(atom.type, None))
        if charge is None:
            raise error.RuntimeError("Could not figure out charge")
        if hasattr(charge, 'rescale'):
            charge = charge.rescale(elementary_charge)
        return float(charge)

    charges = array([get_charge(atom) for atom in structure], dtype='float64')
    positions = array([atom.pos for atom in structure], dtype='float64')
    positions = dot(positions, inv(structure.cell).T)
    reduced_forces = zeros((len(structure), 3), dtype='float64', order='C')
    cartesian_forces = zeros((len(structure), 3), dtype='float64', order='C')
    stress = zeros(6, dtype='float64')

    cdef:
        long c_reduced_forces =  reduced_forces.ctypes.data
        long c_cartesian_forces =  cartesian_forces.ctypes.data
        long c_stress =  stress.ctypes.data
        long c_cell = cell.ctypes.data
        long c_charges = charges.ctypes.data
        long c_positions = positions.ctypes.data
        double energy = 0e0

    ewaldc(verbose, energy, <double*>c_reduced_forces, <double*>c_cartesian_forces,
           <double*>c_stress, len(structure), <double*>c_positions, <double*>c_charges, cutoff,
           <double*>c_cell)

    result = structure.copy()
    for atom, force in zip(result, cartesian_forces):
        atom.force = force * Ry / a0
        result.energy = energy * Ry
        result.stress = zeros((3, 3), dtype='float64') * Ry
        result.stress[0, 0] = stress[0] * Ry
        result.stress[1, 1] = stress[1] * Ry
        result.stress[2, 2] = stress[2] * Ry
        result.stress[0, 1] = stress[3] * Ry
        result.stress[1, 0] = stress[3] * Ry
        result.stress[1, 2] = stress[4] * Ry
        result.stress[2, 1] = stress[4] * Ry
        result.stress[0, 2] = stress[5] * Ry
        result.stress[2, 0] = stress[5] * Ry

    return result
