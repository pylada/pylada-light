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
import numpy as np
cimport numpy as np

cpdef __gvectors(double[:, ::1] cell, double tolerance):
    """ Computes all gvectors in prolate defined by cell """
    from numpy import abs, max, dot, array, sum, square, cross, ceil
    from numpy.linalg import det, norm
    cdef:
        double volume = abs(det(cell))
        double[:] a0 = cell[:, 0]
        double[:] a1 = cell[:, 1]
        double[:] a2 = cell[:, 2]

    lengths = sum(square(cell), axis=0)
    max_norm = max(norm(cell, axis=0))

    cdef:
        int n0 = ceil(max_norm * norm(cross(a1, a2)) / volume)
        int n1 = ceil(max_norm * norm(cross(a2, a0)) / volume)
        int n2 = ceil(max_norm * norm(cross(a0, a1)) / volume)

    gvectors = [], [], []
    for i in range(-n0, n0+1):
        for j in range(-n1, n1+1):
            for k in range(-n2, n2+1):
                g = dot(cell, array([i, j, k], dtype='float64'))
                glength = dot(g, g)
                for length, result in zip(lengths, gvectors):
                    if abs(length - glength) < tolerance:
                        result.append(g)
    return gvectors



cdef __cell_invariants(double[:, ::1] cell, double tolerance):
    from numpy import zeros, identity, abs, dot, array, require, allclose
    from numpy.linalg import det, inv
    from itertools import product

    cdef int ndims = len(cell[0, :])
    result = [require(identity(ndims), dtype='float64', requirements=['F_CONTIGUOUS'])]

    # gvectors contains all vectors in prolate define by lengths
    gvectors = __gvectors(cell, tolerance)

    invcell = inv(cell)
    for gvecs in product(*gvectors):
        rotation = array(gvecs).T
        if abs(det(rotation)) < tolerance:
            continue

        rotation = dot(rotation, invcell)
        if allclose(rotation, identity(ndims), tolerance):
            continue

        # check matrix is a rotation
        if not allclose(dot(rotation, rotation.T), identity(ndims), tolerance):
            continue

        # check rotation not in list 
        doadd = True
        for symop in result:
            if allclose(symop, rotation, tolerance):
                doadd = False
                break
        if doadd:
            result.append(require(rotation, dtype='float64', requirements=['F_CONTIGUOUS']))

    return result


def cell_invariants(cell, tolerance=1e-12):
    """ Finds and stores point group operations

        Rotations are determined from G-vector triplets with the same norm as the unit-cell vectors.
        Implementation taken from ENUM_.

        :param structure:
            The :py:class:`Structure` instance for which to find the space group. Can also be a 3x3
            matrix.

        :param tolerance:
            acceptable tolerance when determining symmetries. Defaults to 1e-8.

        :returns:
            python list of affine symmetry operations for the given structure. Each element is a 4x3
            numpy array, with the first 3 rows forming the rotation, and the last row is the
            translation.  The affine transform is applied as rotation * vector + translation.
            `cell_invariants` always returns rotations (translation is zero).

        .. _ENUM: http://enum.sourceforge.net/
    """
    from numpy import require
    # Makes it easier to input structure 
    cell = getattr(cell, 'cell', cell)
    cell = require(cell, dtype='float64', requirements=['C_CONTIGUOUS'])
    if len(cell.shape) != 2:
        raise ValueError("Expected a matrix as input")
    if cell.shape[0] != cell.shape[1]:
        raise ValueError("Expected a *square* matrix as input")
    if cell.shape[0] != 3:
        raise ValueError("Expected a 3x3 matrix as input")
    return __cell_invariants(cell, tolerance)

def space_group(lattice, tolerance=1e-12):
    """ Finds and stores point group operations

        Implementation taken from ENUM_.

        :param lattice:
            The :class:`Structure` instance for which to find the point group.

        :param tolerance:
            Acceptable tolerance when determining symmetries. Defaults to 1e-8.

        :returns:
            python list of affine symmetry operations for the given structure. Each element is a 4x3
            numpy array, with the first 3 rows forming the rotation, and the last row is the
            translation.  The affine transform is applied as rotation * vector + translation.

        :raises ValueError:
            if the input  structure is not primitive.

        .. _ENUM: http://enum.sourceforge.net
    """
    from numpy import dot, allclose, zeros
    from numpy.linalg import inv
    from pylada.crystal import gruber, Atom, into_voronoi, into_cell
    if len(lattice) == 0:
        raise ValueError("Empty lattice")

    # if not is_primitive(lattice, tolerance)):
    #     raise ValueError("Input lattice is not primitive")

    # Finds minimum translation.
    translation = lattice[0].pos
    cell = gruber(lattice.cell, tolerance=tolerance)
    invcell = inv(cell)

    point_group = cell_invariants(lattice.cell)
    assert len(point_group) > 0

    centered = [Atom(into_cell(u.pos - translation, cell, invcell), u.type) for u in lattice]

    # translations limited to those from one atom type to othe atom of same type
    translations = [u.pos for u in centered if u.type == lattice[0].type]

    result = []
    for pg in point_group:
        for trial in translations:
            # Checks that this is a mapping of the lattice upon itself.
            for unmapped in centered:
                transpos = into_cell(dot(pg, unmapped.pos) + trial, cell, invcell)
                for atom in centered:
                    if atom.type != unmapped.type:
                        continue
                    if allclose(atom.pos, transpos, tolerance):
                        break
                # else executes only no atom is mapping of unmapped
                else:
                    break
            # else executes only if all positions in structures have mapping
            else:
                transform = zeros((len(trial) + 1, len(trial)), dtype='float64', order='F')
                transform[:3, :3] = pg
                transform[3, :] = into_voronoi(
                    trial - dot(pg, translation) + translation, cell, invcell)
                result.append(transform)
                # only one trial translation is possible, so break out of loop early
                break
    return result
