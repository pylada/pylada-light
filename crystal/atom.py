class Atom(object):
    """ Defines an atomic site

        __init__ accepts different kind of input.
          - The position can be given as:
              - the first *three* positional argument
              - as a keyword argument ``position``,
          - The type can be given as:
              - arguments listed after the first three giving the position. A
              list is created to hold references to these arguments.
              - as a keyword argument ``type``.
          - All other keyword arguments become attributes.
            In other words, one could add ``magnetic=0.5`` if one wanted to
            specify the magnetic moment of an atom.

        For instance, the following will create a silicon atom at the origin::

          atom = Atom(0, 0, 0, 'Si')

        Or we could place a iron atom with a magntic moment::

          atom = Atom(0.25, 0, 0.5, 'Si', moment=0.5)

        The ``moment`` keyword will create a corresponding ``atom.moment``
        keyword with a value of 0.5. There are strictly no limits on what kind
        of type to include as attributes. However, in order to work well with
        the rest of Pylada, it is best if extra attributes are pickle-able.

        .. note::

            the position is always owned by the object. Two atoms will not own
            the same position object.  The position given on input is *copied*,
            *not* referenced.  All other attributes behave like other python
            attributes: they are refence if complex objects and copies if a
            basic python type.
    """

    def __init__(self, *args, **kwargs):
        from numpy import array
        super(Atom, self).__init__()

        if len(args) >= 3 and 'pos' in kwargs:
            raise TypeError(
                "Position given through argument and keyword arguments both")
        if len(args) > 3 and 'type' in kwargs:
            raise TypeError(
                "Type given through argument and keyword arguments both")

        if len(args) >= 3:
            self._pos = array(args[:3])
        elif len(args) == 2 or len(args) == 1:
            self._pos = array(args[0])
        elif 'pos' in kwargs:
            self._pos = array(kwargs.pop('pos'))
        else:
            self._pos = array([0., 0., 0.])
        if len(args) == 4:
            self.type = args[3]
        elif len(args) == 2:
            self.type = args[1]
        elif len(args) > 4:
            self.type = args[3:]
        else:
            self.type = kwargs.pop('type', None)
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        from numpy import require
        if not hasattr(value, 'dtype'):
            self._pos = require(value, dtype=self._pos.dtype)
        else:
            self._pos = value

    def __repr__(self):
        """ Dumps atom to string """
        args = [repr(u) for u in self.pos]
        args.append(repr(self.type))
        for k, v in self.__dict__.items():
            if k != '_pos' and k != 'type':
                args.append(str(k) + "=" + repr(v))
        return self.__class__.__name__ + "(" + ", ".join(args) + ")"

    def to_dict(self):
        result = self.__dict__.copy()
        result['pos'] = result.pop('_pos')
        return result
