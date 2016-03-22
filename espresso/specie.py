###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
#  crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################

""" Defines specie specific methods and objects. """
__docformat__ = "restructuredtext en"
from quantities import atomic_mass_unit
from traitlets import HasTraits
from .trait_types import DimensionalTrait


class Specie(HasTraits):
    """ Holds atomic specie information.

        Instances of this object define an atomic specie for Espresso calculations.
        In addition, it may contain element-related information used to build a
        set of high-throughput jobs.
    """
    mass = DimensionalTrait(atomic_mass_unit, default_value=1, allow_none=False,
                            help="Atomic mass of the element")

    def __init__(self, pseudo, **kwargs):
        """ Initializes a specie.

            :param pseudo:
                filename of the associated pseudo
            :param kwargs:
                Any other keyworkd argument is added as an attribute of this object.
        """
        self.pseudo = pseudo
        # sets up other arguments.
        for k, v in kwargs.items():
            setattr(self, k, v)

    def file_exists(self, pseudo_dir=None):
        """ Absolute path to the pseudo if it exists, None otherwise

            Checks in the following order:

            1. `pseudo_dir` is not None, the path points to a file in `pseudo_dir`
            1. ESPRESSO_PSEUDO exists and the path points to a file there
            1. $HOME/espresso/pseudo exists and the path points to a file there
            1. None
        """
        from os.path import exists, join, abspath, expanduser, expandvars, isfile
        from os import environ, getcwd
        if self.pseudo is None:
            return None
        prefixes = [getcwd()]
        if pseudo_dir is not None:
            prefixes.append(expanduser(expandvars(pseudo_dir)))
        if 'ESPRESSO_PSEUDO' in environ:
            prefixes.append(environ['ESPRESSO_PSEUDO'])
        prefixes.append(expanduser(join('~', 'espresso', 'pseudo')))
        for prefix in prefixes:
            path = join(prefix, self.pseudo)
            if exists(path) and isfile(path):
                return path
        return None
