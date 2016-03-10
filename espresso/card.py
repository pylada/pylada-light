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
__all__ = ['Card']
from traitlets import HasTraits, Unicode, CaselessStrEnum, TraitType
from .trait_types import MutableCaselessStrEnum


class Card(HasTraits):
    """ Defines a Pwscf card """
    subtitle = Unicode(None, allow_none=True)
    name = MutableCaselessStrEnum(allow_none=False)

    def __init__(self, name, value=None, subtitle=None):
        from collections import OrderedDict
        super(HasTraits, self).__init__()
        name = str(name).lower()
        if name not in MutableCaselessStrEnum.card_names:
            MutableCaselessStrEnum.card_names.add(name)
        self.name = name
        self.value = value
        self.subtitle = subtitle

    def __repr__(self):
        """ Prints card as should read by Pwscf """
        if self.subtitle is None and self.value is None:
            return self.name.upper()
        elif self.subtitle is None:
            return "%s\n%s" % (self.name.upper(), self.value)
        else:
            return "%s %s\n%s" % (self.name.upper(), self.subtitle, self.value)

    def read(self, stream):
        doing_title = True
        for line in stream:
            title = line.rstrip().lstrip().split()
            if doing_title:
                if len(title) > 0 and title[0].lower() == self.name:
                    doing_title = False
                    if len(title) > 1:
                        self.subtitle = ' '.join(title[1:])
                    self.value = ""
            elif len(title) > 0 and title[0].lower() not in MutableCaselessStrEnum.card_names:
                self.value += line
            elif not doing_title:
                break
