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

# -*- coding: utf-8 -*-
""" Namelist makes it easy to access and modify fortran namelists """
__docformat__ = "restructuredtext en"
__all__ = ['Namelist']
from ..espresso import logger
from traitlets import HasTraits


class InputTransform(object):
    """ Objects that holds a function to transform the ordered dict of a Namelist """

    def __init__(self, method):
        self.method = method


def input_transform(method):
    """ Adds a transform called when creating an ordered dict from a Namelist

        Method decorated with this object are called as the last step of
        :py:method:`Namelist.namelist`. They are meant to transform the ordered dict before
        they are written as fortran namelists.

        The signature for the method should be `method(self, dictionary, **kwargs)`, where `self` is
        the Namelist object, dictionary is the ordered dict to transform, and the keyword arguments
        are those passed on to :py:method:`Namelist.namelist`.
    """
    return InputTransform(method)


class Namelist(HasTraits):
    """ Defines a recursive Pwscf namelist """

    def __init__(self, dictionary=None):
        from f90nml import Namelist as F90Namelist
        super(HasTraits, self).__init__()
        self.__inputs = F90Namelist()
        if dictionary is not None:
            for key, value in dictionary.items():
                setattr(self, key, value)

    def __getattr__(self, name):
        """ Non-private attributes are part of the namelist proper """
        if name[0] == '_' or self.has_trait(name):
            return super(Namelist, self).__getattr__(name)
        try:
            return self.__inputs[name]
        except KeyError as e:
            raise AttributeError(str(e))

    def __setattr__(self, name, value):
        """ Non-private attributes become part of the namelist proper """
        from collections import Mapping
        if name[0] == '_' or self.has_trait(name):
            super(Namelist, self).__setattr__(name, value)
        elif isinstance(value, Mapping) and not isinstance(value, Namelist):
            self.__inputs[name] = Namelist(value)
        elif hasattr(self, name) and name not in self.__inputs:
            super(Namelist, self).__setattr__(name, value)
        else:
            if name in self.__inputs:
                logger.info("Creating new attribute %s in Namelist" % name)
            self.__inputs[name] = value

    def __len__(self):
        """ Number of parameters in namelist """
        n = 0
        for name in self.trait_names():
            value = getattr(self, name)
            if value is not None:
                n += 1
        return len(self.__inputs) + n

    def __delattr__(self, name):
        if name[0] == '_' or self.has_trait(name):
            super(Namelist, self).__delattr__(name)
        else:
            try:
                self.__inputs.pop(name)
            except KeyError as e:
                raise AttributeError(str(e))

    def namelist(self, **kwargs):
        """ Returns a f90nml Namelist object """
        result = self.__inputs.copy()
        for key in list(result.keys()):
            value = result[key]
            if isinstance(value, Namelist):
                result[key] = value.namelist(**kwargs)
            elif value is None:
                result.pop(key)

        for key in self.trait_names():
            value = getattr(self, key)
            if isinstance(value, Namelist):
                result[key] = value.namelist(**kwargs)
            elif value is not None:
                result[key] = value

        for transform in self.__class__.__dict__.values():
            if isinstance(transform, InputTransform):
                logger.debug("Transforming input using method %s" % transform.method.__name__)
                transform.method(self, result, **kwargs)

        return result

    def write(self, filename=None, **kwargs):
        """ Writes namelist to file or string, or stream

            - if filename is None (default), then returns a string containing namelist in fortran
                format
            - if filename is a string, then it should a path to a file
            - otherwise, filename is assumed to be a stream of some sort, with a `write` method

            Keywords are passed on to :py:method:`Namelist.namelist`
        """
        from .misc import write_f90namelist
        return write_f90namelist(self.namelist(**kwargs), stream=filename)

    def read(self, filename, clear=False):
        """ Read input from file """
        from f90nml import read
        from ..misc import local_path
        if clear:
            self.__inputs.clear()

        filename = local_path(filename)
        dictionary = read(str(filename))
        for key, value in dictionary.items():
            setattr(self, key, value)

    def read_string(self, string, clear=False):
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode='w') as file:
            file.write(string)
            file.seek(0)
            return self.read(file.name, clear=clear)

    def clear(self):
        """ Removes all namelist attributes """
        self.__inputs.clear()

    def names(self):
        """ All names of attributes that end up in the namelist """
        from itertools import chain
        for u in chain(self.__inputs, self.trait_names()):
            yield u

    def __repr__(self):
        result = "%s(" % self.__class__.__name__
        attrs = ""
        for k in self.names():
            value = getattr(self, k)
            if k[0] != '_' and value is not None:
                attrs += ", %s=%s" % (k, repr(value))

        return result + attrs[2:] + ")"

    def printattr(self, name='nml'):
        result = ""
        for k in self.names():
            value = getattr(self, k)
            if k[0] != '_' and value is not None:
                result += "%s.%s=%s\n" % (name, k, repr(value))
        return result
