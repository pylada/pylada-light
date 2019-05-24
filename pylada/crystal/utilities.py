def into_cell(position, cell, inverse=None):
    """ Returns Folds vector into periodic the input cell

        :param position:
            Vector/position to fold back into the cell
        :param cell:
            The cell matrix defining the periodicity
        :param invcell:
            Optional. The *inverse* of the cell defining the periodicity. It is
            computed if not given on input.
    """
    from numpy import dot, floor
    from numpy.linalg import inv
    if inverse is None:
        inverse = inv(cell)
    result = dot(inverse, position)
    return dot(cell, result - floor(result + 1e-12))


def zero_centered(position, cell, inverse=None):
    """ Folds vector back to origin

        This may not be the vector with the smallest possible norm if the cell
        is very skewed

        :param position:
            Vector/position to fold back into the cell
        :param cell:
            The cell matrix defining the periodicity
        :param invcell:
            Optional. The *inverse* of the cell defining the periodicity. It is
            computed if not given on input.
    """
    from numpy import dot, floor, abs
    from numpy.linalg import inv
    if inverse is None:
        inverse = inv(cell)
    result = dot(inverse, position)
    result -= floor(result + 0.5 + 1e-12)
    for i in range(result.size):
        if abs(result[i] - 0.5) < 1e-12:
            result[i] = -0.5
        elif result[i] < -0.5:
            result[i] += 1e0
    return dot(cell, result)


def into_voronoi(position, cell, inverse=None):
    """ Folds vector into first Brillouin zone of the input cell

        Returns the periodic image with the smallest possible norm.

        :param position:
            Vector/position to fold back into the cell
        :param cell:
            The cell matrix defining the periodicity
        :param invcell:
            Optional. The *inverse* of the cell defining the periodicity. It is
            computed if not given on input.
    """
    from numpy import dot, floor
    from numpy.linalg import inv, norm
    if inverse is None:
        inverse = inv(cell)
    center = dot(inverse, position)
    center -= floor(center)

    result = center
    n = norm(dot(cell, center))
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            for k in range(-1, 2, 1):
                translated = [i, j, k] + center
                n2 = norm(dot(cell, translated))
                if n2 < n:
                    n = n2
                    result = translated
    return dot(cell, result)


def are_periodic_images(pos0, pos1, invcell=None, cell=None, tolerance=1e-8):
    """ True if two vector are periodic images of one another

        :param pos0:
            First position
        :param pos1:
            Second position
        :param invcell:
            The *inverse* of the cell defining the periodicity
        :param cell:
            The cell defining the periodicity
        :param tolerance:
            Fuzzy tolerance criteria

        Only one of cell and invcell need be given.
    """
    from numpy import dot, floor
    from numpy.linalg import inv, norm
    from .. import error
    if invcell is None:
        if cell is None:
            raise error.ValueError("One of cell or invcell should be given")
        invcell = inv(cell)

    result = dot(invcell, pos0 - pos1)
    result -= floor(result + tolerance)
    return all(abs(result) < tolerance)
