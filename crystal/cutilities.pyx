import numpy as np
cimport numpy as np

cdef extern from "crystal/types.h" namespace "pylada::types":
    ctypedef int t_int

cdef extern from "crystal/smith_normal_form.h" namespace "pylada":
    void snf "pylada::smith_normal_form" (t_int *_S, t_int *_L, t_int *_R) except +

def smith_normal_form(np.ndarray cell not None):
    """ Computes smith normal form on input matrix """
    from numpy.linalg import det
    from numpy import require
    if cell.ndim == 2 and cell.shape[0] != 3 and cell.shape[1] != 3:
        raise ValueError("Can only compute Smith normal form of 3x3 matrix")
    if cell.dtype != 'intc':
        return smith_normal_form(require(cell, dtype='intc'))

    if det(cell) == 0:
        raise ValueError("Input cell is singular")

    left = np.identity(3, dtype='intc')
    right = np.identity(3, dtype='intc')
    S = cell.copy()

    cdef:
        long L_data = left.ctypes.data
        long S_data = S.ctypes.data
        long R_data = right.ctypes.data


    snf(<t_int*>S_data, <t_int*>L_data, <t_int*>R_data)
    return S, left, right
