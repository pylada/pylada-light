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

def map_sites(mapper, mappee, cmp=None, double tolerance=1e-12):
    """ Map sites from a lattice onto a structure

        This function finds out which atomic sites in a supercell refer to the sites in a parent
        lattice. ``site`` attribute are added to the atoms in the ``mappee`` structure. These
        attributes hold an index to the relevant sites in the mapper.  If a particular atom could
        not be mapped, then ``site`` is None

        :param mapper:
            :class:`Structure` instance acting as the parent lattice

        :param mappee:
            :class:`Structure` instance acting as the supercell

        :param cmp:
            Can be set to a callable which shall take two atoms as input and return True if their
            occupation (and other attributes) are equivalent.

        :param tolerance:
            Tolerance criteria when comparing distances

        :returns: True if all sites in mappee where mapped to mapper
    """
    from numpy.linalg import inv, norm
    from numpy import dot, round, allclose, array, argmin, nonzero
    from . import gruber, into_cell, into_voronoi
    from .. import error

    if len(mapper) == 0:
        raise error.ValueError("Empty mapper structure")
    if len(mappee) == 0:
        raise error.ValueError("Empty mappee structure")

    if cmp is None:
        cmp = lambda x, y: x.type == y.type


    cell = gruber(mapper.cell)
    invcell = inv(cell)
    cdef double mapper_scale = mapper.scale
    cdef double scale_ratio = mappee.scale.rescale(mapper.scale.units) / mapper.scale
    cdef double dist_tolerance = tolerance / mapper_scale
    allmapped = True

    intcell = dot(invcell, mappee.cell) * scale_ratio
    if not allclose(intcell, round(intcell + 1e-8), tolerance):
        raise error.ValueError("Mappee not a supercell of mapper")

    sites = array([into_cell(site.pos, cell, invcell) for site in mapper]) / scale_ratio
    for atom in mappee:
        distances = array([norm(into_voronoi(atom.pos - site, cell, invcell)) for site in sites])
        found_sites = nonzero(distances < dist_tolerance)
        if cmp is not None:
            found_sites = [index for index in found_sites if cmp(mapper[index], atom)]
        if len(found_sites) == 0:
            atom.site = None
            allmapped = False
        elif len(found_sites) == 1:
            atom.site = found_sites[0]
        else:
            raise error.RuntimeError("Sites %s are equivalent" % found_sites)

    return allmapped
