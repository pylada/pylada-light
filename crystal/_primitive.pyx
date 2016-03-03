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

cdef __translations(structure, double tolerance):
    """ Looks for internal translations """
    from numpy.linalg import inv
    from numpy import all, abs, allclose
    from . import gruber, into_cell, into_voronoi

    cell = gruber(structure.cell)
    invcell = inv(cell)

    front_type = structure[0].type
    center = structure[0].pos
    translations = []
    for site in structure:
        if front_type != site.type:
            continue

        translation = into_voronoi(site.pos - center, cell, invcell)
        if all(abs(translation) < tolerance):
            continue

        for mapping in structure:
            pos = into_cell(mapping.pos + translation, cell, invcell)
            for mappee in structure:
                if mapping.type == mappee.type and allclose(mappee.pos, pos, tolerance):
                    break
            else:
                break
        else:
            translations.append(into_voronoi(translation, cell, invcell))

    return translations


def primitive(structure, double tolerance=1e-8):
    from numpy.linalg import inv, det
    from numpy import all, abs, array, dot, allclose, round
    from . import gruber, into_cell, into_voronoi, into_cell

    if len(structure) == 0:
        raise ValueError("Empty structure")

    result = structure.copy()
    cell = gruber(result.cell)
    invcell = inv(cell)
    for atom in result:
        atom.pos = into_cell(atom.pos, cell, invcell)

    translations = __translations(result, tolerance)
    if len(translations) == 0:
        return result

    # adds original translations.
    translations.append(cell[:, 0])
    translations.append(cell[:, 1])
    translations.append(cell[:, 2])

    # Looks for cell with smallest volume 
    new_cell = result.cell.copy()
    volume = abs(det(new_cell))
    for i, first in enumerate(translations):
        for j, second in enumerate(translations):
            if i == j:
                continue
            for k, third in enumerate(translations):
                if i == k or j == k:
                    continue
                trial = array([first, second, third]).T
                if abs(det(trial) < 1e-12):
                    continue
                if abs(det(trial)) > volume - 3.0 * tolerance:
                    continue

                if det(trial) < 0e0:
                    trial[:, 2] = second
                    trial[:, 1] = third
                    if det(trial) < 0e0:
                        raise RuntimeError("Negative volume")
                integer_cell = dot(inv(trial), cell)
                if allclose(integer_cell, round(integer_cell + 1e-7), 1e-8):
                    new_cell = trial
                    volume = abs(det(trial))

    # Found the new cell with smallest volume (e.g. primivite)
    if abs(structure.volume - volume) < tolerance:
        raise RuntimeError("Found translation but no primitive cell.")

    # now creates new lattice.
    result.clear()
    result.cell = gruber(new_cell)
    invcell = inv(result.cell)
    for site in structure:
        pos = into_cell(site.pos, result.cell, invcell)
        for unique in result:
            if site.type == unique.type and allclose(unique.pos, pos, tolerance):
                break
        else:
            result.append(site.copy())
            result[-1].pos = pos

    if len(structure) % len(result) != 0:
        raise RuntimeError("Nb of atoms in output not multiple of input.")

    if abs(len(structure) * result.volume - len(result) * structure.volume) > tolerance:
        raise RuntimeError("Size and volumes do not match.")

    return result;

def is_primitive(structure, double tolerance = 1e-12):
    """ True if the lattice is primitive

        :param lattice:
            :class:`Structure` for which to get primitive unit-cell lattice. Cannot be empty.
            Must be deepcopiable.

        :param tolerance:
            Tolerance when comparing positions
    """
    from . import into_cell, gruber
    from numpy.linalg import inv
    if len(structure) == 0:
        raise ValueError("Empty structure")

    result = structure.copy()
    cell = gruber(result.cell)
    invcell = inv(cell)
    for atom in result:
        atom.pos = into_cell(atom.pos, cell, invcell)

    return len(__translations(result, tolerance)) == 0
