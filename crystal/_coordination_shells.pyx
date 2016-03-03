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
cdef __natoms(int natoms, int nshells):
    """ Number of atoms for computing coordination """
    from numpy import sum
    if natoms > 0:
        return natoms
    # based on fcc
    cdef int result = sum([12, 6, 24, 12, 24, 8, 48, 6, 32][:nshells])
    if nshells < 12:
        result += 6
    else:
        result *= 8

    return result

def coordination_shells(structure, int nshells, center, tolerance=1e-12, natoms=0):
    """ Creates list of coordination shells up to given order

        :param structure:
            :class:`Structure` from which to determine neighbors

        :param nshells:
            Integer number of shells to compute. More may be found.

        :param center:
            Position for which to determine first neighbors

        :param tolerance:
            Tolerance criteria for judging equidistance

        :param natoms:
            Integer number of neighbors to consider. Defaults to fcc + some security.
            If too small, then the shells may be incomplete, especially strangely shaped cells.

        :returns:
            A list of lists of tuples. The outer list is over coordination shells.  The inner list
            references the atoms in a shell.  Each innermost tuple contains a reference to the atom
            in question, a vector from the center to the relevant periodic image of the atom, and
            finally, the associated distance.
    """
    from bisect import bisect_left
    from itertools import product
    from numpy.linalg import inv, norm
    from numpy import power, max, cross, dot, ceil
    from . import gruber, into_voronoi

    cdef int N = len(structure)
    cell = gruber(structure.cell, 3e0*tolerance)
    invcell = inv(cell)
    cdef double volume = structure.volume
    cdef int maxatoms = __natoms(natoms, nshells)
    center = getattr(center, 'pos', center)

    # Finds out how far to look.
    max_norm = max(norm(cell, axis=0))
    cdef double r = power(max([1, float(maxatoms)]) / float(N), 1e0/3e0)
    cdef int n0 = max([1.0, ceil(r * max_norm * norm(cross(cell[:, 1], cell[:, 2])) / volume)])
    cdef int n1 = max([1.0, ceil(r * max_norm * norm(cross(cell[:, 2], cell[:, 0])) / volume)])
    cdef int n2 = max([1.0, ceil(r * max_norm * norm(cross(cell[:, 0], cell[:, 1])) / volume)])
    while n0 * n1 * n2 * 8 * N < maxatoms:
        n0 += 1
        n1 += 1
        n2 += 1

    cdef double max_distance = 1.2 * power(volume / float(maxatoms), 2e0/3e0) \
            * float(maxatoms * maxatoms)
    result, distances = [], []
    for atom in structure:
        start = into_voronoi(atom.pos - center, cell, invcell)
        if dot(start, start) > max_distance:
            continue
        for translation in product(range(-n0, n0+1), range(-n1, n1 + 1), range(-n2, n2 + 1)):
            pos = start + dot(cell, translation)
            distance = norm(pos)
            # find position in sorted list
            i = bisect_left(distances, distance)
            # add to new shell 
            if len(distances) == i:
                distances.append(distance)
                result.append([(atom, pos, distance)])
            # add to old shell
            elif len(distances) > i and abs(distance - distances[i]) < tolerance:
                result[i].append((atom, pos, distance))
            elif i > 0 and abs(distance - distances[i - 1]) < tolerance:
                result[i - 1].append((atom, pos, distance))
            else:
                distances.insert(i, distance)
                result.insert(i, [(atom, pos, distance)])
            if len(distances) > nshells + 2:
                distances = distances[:nshells + 2]
                result = result[:nshells + 2]

    if len(result) <= nshells + 1:
        return coordination_shells(structure, nshells, center, tolerance=1e-12, natoms=maxatoms * 2)
    # drop last shell since it is most likely to be incomplete 
    return result[1:-1]
