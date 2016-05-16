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
import six
from traitlets import TraitType, Unicode, CaselessStrEnum, Enum
__all__ = ['CardNameTrait', 'DimensionalTrait', 'dimensional_trait_as_other', 'String',
           'LowerCaseString', 'CaselessStringEnum']


class CardNameTrait(TraitType):
    """ CaselessStringEnum to which we can easily add allowed values """

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
        from traitlets import TraitError
        if hasattr(value, 'simplified'):
            if value.simplified.dimensionality != self.units.simplified.dimensionality:
                raise TraitError(
                    "Input units (%s) are indimensional with (%s)"
                    % (value.units, self.units)
                )
            return value
        elif value is not None:
            return value * self.units
        else:
            return value


def dimensional_trait_as_other(name, units=None, other=float):
    """ Dimensional traits must be transformed to floating point for fortran

        This function returns an input_transform function that makes sure the input value is
        transformed by first rescaling to the given units, then transforming to float.

        :param name: str
            Key to look for in the namelist dictionary
        :units: quantities.UnitQuantity, None, DimensionalTrait
            Rescale to these units
        :other: callable
            Transforms dictionary item (possibly rescaled) to fortran input
    """
    from .namelists import input_transform

    if hasattr(units, 'units'):
        units = units.units

    @input_transform
    def __dimensional_input_transform(self, dictionary, **kwargs):
        value = dictionary.get(name, None)
        if value is None:
            return
        if hasattr(value, 'rescale') and units is not None:
            value = value.rescale(units)
        dictionary[name] = other(value)
    __dimensional_input_transform.__doc__ = """ Tranforms %s for writing to namelist """

    return __dimensional_input_transform


if six.PY2:
    class String(TraitType):
        """ String trait """

        default_value = None
        info_text = 'A string'

        def validate(self, obj, value):
            if value is None:
                return None
            return str(value)

    class CaselessStringEnum(Enum):
        """An enum of strings where the case should be ignored."""

        def __init__(self, values, default_value=None, **metadata):
            values = [str(value) for value in values]
            super(CaselessStringEnum, self).__init__(values, default_value=default_value, **metadata)

        def validate(self, obj, value):
            value = str(value)
            for v in self.values:
                if v.lower() == value.lower():
                    return v
            self.error(obj, value)
else:
    String = Unicode
    CaselessStringEnum = CaselessStrEnum


class LowerCaseString(String):
    """ String that are always lowercase """

    default_value = None
    info_text = 'A lowercase string'

    def validate(self, obj, value):
        if value is None:
            return None
        return super(LowerCaseString, self).validate(obj, value).lower()
