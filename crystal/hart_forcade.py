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
class HFTransform(object):
    """ The Hart-Forcade transform computes the cyclic group of supercell

        with respect to its backbone lattice. It can then be used to index atoms in the supercell,
        irrespective of which periodic image is given [HF]_.
    """
    def __init__(self, lattice, supercell):
        """ Creates a Hart-Forcard transform

            :param lattice:
                Defines the cyclic group
            :type lattice:
                :py:func:`Structure` or matrix

            :param supercell:
                Supercell for which to compute cyclic group
            :type supercell:
                :py:func:`Structure` or matrix
        """
        from numpy import require, dot, floor, allclose, round, array
        from numpy.linalg import inv
        from . import smith_normal_form
        from .. import error
        cell = require(getattr(lattice, 'cell', lattice), dtype='float64')
        supercell = require(getattr(supercell, 'cell', supercell), dtype='float64')

        invcell = inv(cell)
        invsupcell = dot(invcell, supercell)
        integer_cell = round(invsupcell).astype('intc')
        if not allclose(floor(invsupcell + 0.01), invsupcell, 1e-8):
            raise error.ValueError("second argument is not a supercell of first argument")

        snf, left, right = smith_normal_form(integer_cell)
        self.transform = dot(left.astype('float64'), invcell)
        self.quotient = array([snf[i, i] for i in range(cell.shape[0])], dtype='intc')

    def flatten_indices(self, i, j, k, site=0):
        """ Flattens cyclic Z-group indices

            :param int i:
                First index into cyclic Z-group

            :param int j:
                Second index into cyclic Z-group

            :param int j:
                Third index into cyclic Z-group

            :param int site:
                Optional site index for multilattices

            :returns: An integer which can serve as an index into a 1d array
        """
        return k  + self.quotient[2] * (j + self.quotient[1] * (i + site * self.quotient[0]))

    def index(self, pos, site=0):
        """ Flat index into cyclic Z-group

            :param pos: (3d-vector)
                Atomic position with respect to the sublattice of interest.  Do not forget to shift
                the sublattice back to the origin.

            :param site: (integer)
                Optional site index. If there are more than one sublattice in the structure, then
                the flattened indices need to take this into account.
        """
        return self.flatten_indices(*self.indices(pos), site=site)

    def indices(self, pos):
        """ indices of input atomic position in cyclic Z-group

            :param pos:
                A 3d-vector in the sublattice of interest

            :returns:
                The 3-d indices in the cyclic group
        """
        from numpy import dot, zeros, round, allclose
        from .. import error
        if len(pos) != len(self.quotient):
            raise error.ValueError("Incorrect vector size")
        pos = dot(self.transform, pos)
        integer_pos = round(pos + 1e-8).astype('intc')
        if not allclose(integer_pos, pos, 1e-12):
            raise error.ValueError("Position is not on the lattice")
        result = zeros(len(pos), dtype='intc')
        for i in range(len(pos)):
            result[i] = integer_pos[i] % self.quotient[i]
            if result[i] < 0:
                result[i] += self.quotient[i]
        return result
