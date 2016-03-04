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

__docformat__ = "restructuredtext en"
from libcpp cimport bool
cimport numpy
cimport cython


cdef extern from "cmath" namespace "std":
    float floor(float)
    double floor(double)


cpdef bool _is_integer(cython.floating[:, ::1] vector, cython.floating tolerance):
    cdef:
        int ni = vector.shape[0]
        int nj = vector.shape[1]
        int i, j

    for i in range(ni):
        for j in range(nj):
            if abs(floor(vector[i, j] + 1e-3) - vector[i, j]) > tolerance:
                return False

    return True


cpdef int _lexcompare(cython.integral[::1] a, cython.integral[::1] b):
    """ Lexicographic compare of two numpy arrays

        Compares two arrays, returning 1 or -1 depending on the first element that is not equal, or
        zero if the arrays are identical.

        :returns:
            - a > b: 1
            - a == b: 0
            - a < b: -1

    """
    from .. import error
    if len(a) != len(b):
        raise error.ValueError("Input arrays have different length")

    cdef:
        int na = len(a)
        int i

    for i in range(na):
        if a[i] < b[i]:
            return -1
        elif a[i] > b[i]:
            return 1

    return 0


cdef class NDimIterator(object):
    """ Defines an N-dimensional iterator

        The two following loop are mostly equivalent:

            >>> for x in NDimIterator(1, 5, 6): print x
            >>> from itertools import product
            >>> for x in product(range(1, 2), range(1, 5), range(1, 7)): print x

        The main differences are:

            1. :py:class:`NDimIterator` yields a numpy array
            1. :py:class:`NDimIterator` always yield the same numpy array, to avoid memory
                reallocation
            1. :py:class:`NDimIterator` cannot be used  with zip_ and similar functions
    """
    cdef:
        int __length
        int [::1] __limits
        int [::1] __current
        object limits
        object current


    def __init__(self, *args):
        from numpy import array, ones
        self.limits = array(args, dtype='intc').flatten()
        if any(self.limits < 0):
            raise ValueError("Can't use negative values in NdimIterator")
        self.__limits = self.limits
        self.current = ones(self.limits.shape, dtype='intc')
        self.current[-1] = 0
        self.__current = self.current
        self.__length = len(self.limits)


    def __iter__(self):
        return self


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def __next__(self):
        cdef int i
        for i in range(1, self.__length + 1):
            if self.__current[self.__length - i] == self.__limits[self.__length - i]:
                self.__current[self.__length - i] = 1
            else:
                self.__current[self.__length - i] += 1
                break
        else:
            raise StopIteration("End of NdimIterator loop")

        return self.current

cdef class FCIterator(NDimIterator):
    """ Binary fixed concententration iterator

        Iterates over all binary strings for a given length and fixed number of 1.
    """
    cdef:
        bool __is_first
        object yielded

    def __init__(self, length, ones):
        """ Constructs an iterator over binary strings with fixed concentration """
        from numpy import zeros
        from .. import error
        if length < 0:
            raise ValueError("Negative length")
        if ones < 0:
            raise ValueError("Negative number of 1s")
        if length < ones:
            raise ValueError("More one than bitstrings")
        NDimIterator.__init__(self, *range(length - ones, length))

        self.yielded = zeros(length, dtype='bool_')
        self.reset()


    def __next__(self):
        if self.__is_first:
            self.__is_first = False
            return self.yielded

        cdef int i
        for i in range(self.__length - 1):
            if self.__current[i] != i:
                self.__current[i] -= 1
                break
            elif self.__current[i + 1] != i + 1:
                self.__current[i] = self.__current[i + 1] - 2
        else:
            if self.__current[self.__length - 1] != self.__length - 1:
                self.__current[self.__length - 1] -= 1
            else:
                raise StopIteration("End of FCIterator loop")


        # setup current bitstring
        self.yielded[:] = 0
        for i in range(self.__length):
            self.yielded[self.__current[i]] = 1
        return self.yielded

    def reset(self):
        """ Resets iterator to starting point """
        self.current[:] = self.limits[:]
        self.yielded[len(self.yielded) - self.__length:] = 1
        self.yielded[:len(self.yielded) - self.__length] = 0
        self.__is_first = True
