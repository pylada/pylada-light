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
""" Sub-package containing the espresso functional. """
__docformat__ = "restructuredtext en"
__all__ = ['Namelist']
from traitlets import HasTraits


class Namelists(HasTraits):
    """ Defines a recursive Pwscf namelist """

    def __init__(self, dictionary=None):
        from collections import OrderedDict
        super(HasTraits, self).__init__()
        self.__inputs = OrderedDict()
        if dictionary is not None:
            for key, value in dictionary.items():
                setattr(self, key, value)

    def __getattr__(self, name):
        """ Non-private attributes are part of the namelist proper """
        if name[0] == '_':
            return super(Namelists, self).__getattr__(name)
        try:
            return self.__inputs[name]
        except KeyError as e:
            raise AttributeError(str(e))

    def __setattr__(self, name, value):
        """ Non-private attributes become part of the namelist proper """
        from collections import Mapping
        if name[0] == '_':
            super(Namelists, self).__setattr__(name, value)
        elif isinstance(value, Mapping) and not isinstance(value, Namelists):
            self.__inputs[name] = Namelists(value)
        else:
            self.__inputs[name] = value

    def __len__(self):
        """ Number of parameters in namelist """
        return len(self.__inputs)

    @property
    def ordered_dict(self):
        from collections import OrderedDict
        result = self.__inputs.copy()
        for key, value in result.items():
            if isinstance(value, Namelists):
                result[key] = value.ordered_dict
        return result
