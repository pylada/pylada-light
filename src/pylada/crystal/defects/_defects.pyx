cdef extern from "pylada/crystal/defects/third_order.h" namespace "pylada":
    double __third_order "pylada::third_order"(const double *matrix, int n)

def third_order(matrix, int n):
    """ Helps compute third-order charge correction """
    from numpy import require
    matrix = require(matrix, dtype='float64', requirements=['C_CONTIGUOUS'])
    cdef long cdata = matrix.ctypes.data
    return __third_order(<double*>cdata, n)
