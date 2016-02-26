from collections import MutableSequence


class Structure(MutableSequence):
    """ Defines a crystal structure

        A structure is a special kind of sequence containing only
        :class:`Atom`. It also sports attributes such a cell and scale.
    """

    def __init__(self, *args, **kwargs):
        """ Creates a structure

            - if 9 numbers are given as arguments, these create the cell
              vectors, where the first three numbers define the first row of
              the matrix (not the first cell vector, eg column).
            - if only one object is given it should be the cell matrix
            - the cell can also be given as a keyword argument
            - the scale can only be given as a keyword argument
            - All other keyword arguments become attributes.  In other words,
              one could add ``magnetic=0.5`` if one wanted to specify the
              magnetic moment of a structure. It would later be accessible as
              an attribute, eg as ``structure.magnetic``.

            .. note:: The cell is always owned by the object. Two structures
                will not own the same cell object.  The cell given on input is
                *copied*, *not* referenced. All other attributes behave like
                other python attributes: they are refence if complex objects
                and copies if a basic python type.
        """
        from quantities import angstrom
        from numpy import identity, array, all

        if len(args) == 9 or len(args) == 3:
            self._cell = array(args, dtype='float64').reshape(3, 3)
        elif len(args) == 4 or len(args) == 2:
            self._cell = array(args, dtype='float64').reshape(2, 2)
        elif len(args) == 1:
            self._cell = array(args, dtype='float64')
        elif len(args) != 0:
            raise TypeError(
                "Incorrect number of arguments when creating structure")

        if '_cell' in self.__dict__ and 'cell' in kwargs:
            raise TypeError("Cell given as argument and keyword argument")
        elif '_cell' not in self.__dict__:
            self._cell = array(kwargs.pop(
                'cell', identity(3)), dtype='float64')

        self._atoms = []
        self._scale = kwargs.pop('scale', 1e0 * angstrom)
        if not hasattr(self._scale, 'units'):
            self._scale *= angstrom

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def cell(self):
        """ Cell matrix in cartesian coordinates

            Unlike most ab-initio codes, cell-vectors are given in column
            vector format. The cell does not yet have units. Units depend upon
            :class:`Structure.scale`. Across pylada, it is expected that a
            cell time this scale are angstroms. Finally, the cell is owned
            internally by the structure. It cannot be set to reference an
            object (say a list or numpy array). 

            ``structure.cell = some_list`` will copy the values of ``some_list``.
        """
        return self._cell

    @cell.setter
    def cell(self, cell):
        from numpy import require
        if not hasattr(cell, 'dtype'):
            self._cell = require(cell, dtype=self._cell.dtype)
        else:
            self._cell = cell

    @property
    def scale(self, scale):
        """ Scale factor of this structure

            Should be a number or unit given by the python package `quantities
            <http://packages.python.org/quantities/index.html>`_. If given as a
            number, then the current units are kept. Otherwise, it changes the
            units.
        """
        if hasattr(scale, 'units'):
            self._scale = scale
        elif hasattr(self._scale, 'units'):
            self._scale = scale * self._scale.units
        else:
            self._scale = scale

    @scale.getter
    def scale(self):
        return self._scale

    def __iter__(self):
        """ Iterates over atoms """
        return self._atoms.__iter__()

    def __len__(self):
        """ Number of atoms in structure """
        return len(self._atoms)

    def append(self, *args, **kwargs):
        from .atom import Atom
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], Atom):
            self._atoms.append(args[0])
        else:
            self._atoms.append(Atom(*args, **kwargs))

    def add_atom(self, *args, **kwargs):
        """ Adds atom to structure

            The argument to this function is either another atom, in which case
            a reference to that atom is appended to the structure. Or, it is
            any arguments used to initialize atoms in :class:`Atom`. Finally,
            this function can be chained as follows::

              structure.add_atom(0,0,0, 'Au') \
                       .add_atom(0.25, 0.25, 0.25, ['Pd', 'Si'], m=5)\
                       .add_atom(atom_from_another_structure)

            In the example above, both ``structure`` and the *other* structure
            will reference the same atom (``atom_from_another_structure``).
            Changing, say, that atom's type in one structure will also change
            it in the other.

            :returns: The structure itself, so that add_atom methods can be chained.
        """
        self.append(*args, **kwargs)
        return self

    def extend(self, args):
        """ Adds atoms to structure """
        from collections import Sequence
        from .atom import Atom
        for arg in args:
            if isinstance(arg, Atom):
                self._atoms.append(arg)
            elif isinstance(arg, Sequence):
                self._atoms.append(Atom(*arg))
            else:
                raise TypeError("Cannot convert argument to Atom")

    def insert(self, index, *args, **kwargs):
        """ Insert atoms in given position """
        from .atom import Atom
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], Atom):
            atom = args[0]
        else:
            atom = Atom(*args, **kwargs)
        self._atoms.insert(index, atom)

    def __setitem__(self, index, atom):
        from collections import Sequence
        from .atom import Atom
        if isinstance(atom, Sequence):
            for a in atom:
                if not isinstance(a, Atom):
                    raise ValueError("Input should be of type Atom")
        else:
            if not isinstance(atom, Atom):
                raise ValueError("Input should be of type Atom")

        return self._atoms.__setitem__(index, atom)

    def __getitem__(self, index):
        return self._atoms.__getitem__(index)

    def __delitem__(self, index):
        return self._atoms.__delitem__(index)

    def clear(self):
        """ Removes all atoms """
        self._atoms = []

    def pop(self, index=-1):
        """ Removes and returns atom at given position """
        return self._atoms.pop(index)

    def to_dict(self):
        """ Dictionary with shallow copies of items """
        result = {'cell': self.cell, 'scale': self.scale}
        for key, value in self.__dict__.items():
            if key[0] != '_':
                result[key] = value
        for i, atom in enumerate(self):
            result[i] = atom.to_dict()
        return result

    def __repr__(self):
        """ Parsable representation of this object """
        result = self.__class__.__name__ + "("
        cell = self.cell.copy()
        cell = ", ".join([repr(u) for u in cell.reshape(cell.size)])
        result += "%s, scale=%s" % (cell, repr(self.scale))
        for key, value in self.__dict__.items():
            if key[0] != '_':
                result += ", %s=%s" % (key, repr(value))
        result += ")"
        for atom in self._atoms:
            pos = ", ".join([repr(u) for u in atom.pos])
            result += "\\\n" + "    .add_atom(%s, %s" % (pos, repr(atom.type))
            for key, value in atom.__dict__.items():
                if key[0] != '_' and key != 'type':
                    result += ", %s=%s" % (key, repr(value))
            result += ")"
        return result

    def transform(self, rotation, translation=None):
        """ Applies rotation and translation to structure """
        from numpy import dot
        self.cell = dot(rotation, self.cell)
        if translation is None:
            for atom in self:
                atom.pos = dot(rotation, atom.pos)
        else:
            for atom in self:
                atom.pos = dot(rotation, atom.pos) + translation

    def copy(self):
        """ Returns a deepcopy of this structure """
        from copy import deepcopy
        return deepcopy(self)
