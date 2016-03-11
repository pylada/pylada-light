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
""" Pwscf functional """
__docformat__ = "restructuredtext en"
__all__ = ['Pwscf']
from ..espresso import logger
from quantities import bohr_radius
from traitlets import HasTraits, CaselessStrEnum, Unicode, Integer, Instance
from .trait_types import DimensionalTrait
from . import Namelist
from .card import Card


def alias(method):
    """ Helps create a name alias for a given property or attribute """
    if hasattr(method, '__get__'):
        return property(method.__get__,
                        getattr(method, '__set__', None),
                        getattr(method, '__del__', None),
                        doc=getattr(method, 'help', getattr(method, '__doc__', None)))
    else:
        return property(lambda x: getattr(x, method),
                        lambda x, v: setattr(x, method, v),
                        lambda x: delattr(x, method),
                        doc="Alias for %s" % method)


class Control(Namelist):
    """ Control namelist """
    calculation = CaselessStrEnum(['scf', 'nscf', 'bands', 'relax', 'md', 'vc-relax', 'vc-md'],
                                  'scf', allow_none=False, help="Task to be performed")
    title = Unicode(None, allow_none=True, help="Title of the calculation")
    verbosity = CaselessStrEnum(['high', 'low'], 'low', allow_none=False,
                                help="How much talk from Pwscf")
    prefix = Unicode(None, allow_none=True, help="Prefix for output files")


class System(Namelist):
    """ System namelist """
    nbnd = Integer(default_value=None, allow_none=True, help="Number of bands")


class Electrons(Namelist):
    """ Electrons namelist """
    electron_maxstep = Integer(default_value=None, allow_none=True,
                               help="Maximum number of scf iterations")
    itermax = alias(electron_maxstep)


class Pwscf(HasTraits):
    """ Wraps up Pwscf in python """
    control = Instance(Control, args=(), kw={}, allow_none=False)
    system = Instance(System, args=(), kw={}, allow_none=False)
    electrons = Instance(Electrons, args=(), kw={}, allow_none=False)
    k_points = Instance(Card, args=('K_POINTS',), kw={'subtitle': 'gamma'}, allow_none=False,
                        help="Defines the set of k-points for the calculation")

    kpoints = alias(k_points)

    def __init__(self, **kwargs):
        from . import Namelist
        super(Pwscf, self).__init__(**kwargs)
        self.__namelists = Namelist()
        self.__cards = {}

    def write(self, filename=None):
        """ Writes Pwscf input

            - if filename is None (default), then returns a string containing namelist in fortran
                format
            - if filename is a string, then it should a path to a file
            - otherwise, filename is assumed to be a stream of some sort, with a `write` method
        """
        from f90nml import Namelist as F90Namelist
        from os.path import expanduser, expandvars
        from io import StringIO
        if filename is None:
            result = StringIO()
            self.write(result)
            result.seek(0)
            return result

        if isinstance(filename, str):
            path = abspath(expanduser(expandvars(filename)))
            logger.info("%s: writing to file %s", self.__class__.__name__, path)
            with open(path, 'w') as file:
                self.write(file)
            return

        # write namelists first
        namelist = self.__namelists.copy()
        for key, value in self.traits().items():
            if isinstance(value, Namelist):
                namelist[key] = value
        namelist.write(filename)

        # then write cards
        for name in self.traits().keys():
            value = getattr(self, name)
            if isinstance(value, Card):
                value.write(filename)

        for value in self.__cards.values():
            value.write(filename)

    def read(self, filename, clear=True):
        """ Read from a file """
        from os.path import expanduser, expandvars, abspath
        from f90nml import Namelist as F90Namelist
        from .trait_types import CardNameTrait
        from .card import read_cards

        # read namelists first
        if clear:
            self.__namelists.clear()
            self.__cards = {}
            for name in self.trait_names():
                value = getattr(self, name)
                if hasattr(value, 'clear'):
                    value.clear()

        filename = abspath(expandvars(expanduser(filename)))
        logger.info("%s: Reading from file %s", self.__class__.__name__, filename)
        namelist = self.__namelists.read(filename)

        traits = set(self.trait_names()).intersection(self.__namelists.names())
        for traitname in traits:
            newtrait = getattr(self.__namelists, traitname)
            delattr(self.__namelists, traitname)
            trait = getattr(self, traitname)
            for key in newtrait.names():
                setattr(trait, key, getattr(newtrait, key))

        # Then read all cards
        for card in read_cards(filename):
            if card.name in self.trait_names():
                getattr(self, card.name).subtitle = card.subtitle
                getattr(self, card.name).value = card.value
            else:
                self.__cards[card.name] = card

    def __getattr__(self, name):
        """ look into extra cards and namelists """
        if name in self.__cards:
            return self.__cards[name]
        elif hasattr(self.__namelists, name):
            return getattr(self.__namelists, name)
        return super(Pwscf, self).__getattr__(name)

    def add_card(self, name, value=None, subtitle=None):
        """ Adds a new card, or sets the value of an existing one """
        if isinstance(getattr(self, name, None), Card):
            card = getattr(self, name)
        else:
            logger.info("%s: Adding new card %s", self.__class__.__name__, name)
            card = Card(name)
            self.__cards[name] = card
        card.subtitle = subtitle
        card.value = value

    def add_namelist(self, name, dictionary=None):
        """ Adds a new namelist, or sets the value of an existing one """
        from .namelist import Namelist
        if isinstance(getattr(self, name, None), Namelist):
            namelist = getattr(self, name)
            namelist.clear()
        else:
            logger.info("%s: Adding new namelist %s", self.__class__.__name__, name)
            namelist = Namelist()
            self.__namelists[name] = namelist
        if dictionary is not None:
            for key, value in dictionary.items():
                setattr(namelist, key, value)
