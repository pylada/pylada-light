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
from traitlets import HasTraits, CaselessStrEnum, TraitType
from .trait_types import CardNameTrait, LowerCaseUnicode
from ..espresso import logger


class Card(HasTraits):
    """ Defines a Pwscf card """
    subtitle = LowerCaseUnicode(None, allow_none=True)
    name = CardNameTrait(allow_none=False)

    def __init__(self, name, value=None, subtitle=None):
        from collections import OrderedDict
        super(HasTraits, self).__init__()
        name = str(name).lower()
        if name not in CardNameTrait.card_names:
            CardNameTrait.card_names.add(name)
        self.name = name
        self.value = value
        self.subtitle = subtitle

    def __str__(self):
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
                    self.value = None
            elif len(title) > 0 and title[0].lower() not in CardNameTrait.card_names:
                if self.value is None:
                    self.value = line.rstrip().lstrip()
                else:
                    self.value += "\n" + line.rstrip().lstrip()
            elif not doing_title:
                break
        else:
            logger.warn("Card %s could not find itself when reading input" % self.name)


def read_cards(stream):
    """ Returns list of cards read from file

        `stream` can be a stream of a path (string). This funtion will avoid namelists, and read
        all cards, as defined by the :py:attr:`CardNameTrait.card_names`.
    """
    from os.path import expandvars, expanduser
    if isinstance(stream, str):
        path = expandvars(expanduser(stream))
        logger.info("Reading cards from %s", path)
        return read_cards(open(path, 'r'))

    results = []
    in_namelist = False
    for line in stream:
        title = line.rstrip().lstrip().split()
        logger.log(5, "line: %s", line[:-1])
        if in_namelist:
            if line.rstrip().lstrip() == '/':
                in_namelist = False
            continue
        elif len(title) == 0:
            continue
        elif title[0][0] == '&':
            in_namelist = True
            continue
        elif title[0].lower() in CardNameTrait.card_names:
            doing_title = False
            subtitle = None
            if len(title) > 1:
                subtitle = ' '.join(title[1:])
            results.append(Card(title[0], subtitle=subtitle, value=None))
        elif len(results) > 0 and title[0].lower() not in CardNameTrait.card_names:
            if results[-1].value is None:
                results[-1].value = line.rstrip().lstrip()
            else:
                results[-1].value += "\n" + line.rstrip().lstrip()

    logger.debug("Read %i cards: %s", len(results), [r.name for r in results])
    return results
