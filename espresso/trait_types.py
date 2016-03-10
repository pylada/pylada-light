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
""" Sub-package containing specialized traitets """
from traitlets import TraitType


class CardNameTrait(TraitType):
    """ CaselessStrEnum to which we can easily add allowed values """

    info_text = 'Pwscf cards'
    card_names = {'atomic_species', 'atomic_positions', 'k_points', 'cell_parameters',
                  'occupations', 'constraints', 'atomic_forces'}
    default_value = next(iter(card_names))

    def validate(self, obj, value):
        value = str(value).lower()
        if value not in CardNameTrait.card_names:
            self.error(obj, "Card name is not one of %s" % CardNameTrait.card_names)
        return value


class DimensionalTrait(TraitType):
    """ Traits with a physical dimension """

    info_text = 'Traits with a physical dimension'

    def __init__(self, units, **kwargs):
        super(DimensionalTrait, self).__init__(**kwargs)
        self.units = units

    def validate(self, obj, value):
        if hasattr(value, 'rescale'):
            return value.rescale(self.units)
        else:
            return value * self.units
