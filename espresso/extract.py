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
""" Pwscf Extraction """
from ..espresso import logger
from ..tools import make_cached
logger = logger.getChild('extract')


class Extract(object):
    """ Extracts stuff from pwscf output """

    def __init__(self, directory=None, prefix='pwscf'):
        from py.path import local
        super(Extract, self).__init__()
        self.directory = str(local() if directory is None else directory)
        """ Directory where files are to be found """
        self.prefix = prefix
        """ Prefix for files and subdirectory """

    @property
    def is_running(self):
        """ True if program is running on this functional. 

            A file '.pylada_is_running' is created in the output folder when it is
            set-up to run Pwscf. The same file is removed when Pwscf returns
            (more specifically, when the :py:class:`pylada.process.ProgramProcess` is
            polled). Hence, this file serves as a marker of those jobs which are
            currently running.
        """
        return self.abspath.join('.pylada_is_running').check(file=True)

    @property
    def abspath(self):
        """ Absolute path to directory """
        from os.path import expandvars
        from py.path import local
        return local(expandvars(str(self.directory)), expanduser=True)

    def __grep_pwscf_out(self, regex):
        from .. import error
        from re import search
        path = self.abspath.join(self.prefix + ".out")
        if not path.check(file=True):
            raise error.IOError("File %s does not exist" % path)
        return search(regex, path.open("rb").read())

    @property
    def success(self):
        """ True if calculation is successful """
        try:
            return self.__grep_pwscf_out(b"JOB DONE.") is not None
        except:
            return False

    @property
    @make_cached
    def functional(self):
        """ Functional used in calculation """
        from .functional import Pwscf
        path = self.abspath.join("%s.in" % self.prefix)
        if not path.check(file=True):
            raise IOError("Could not find input file %s" % path)
        pwscf = Pwscf()
        pwscf.read(str(path))
        return pwscf

    @property
    @make_cached
    def initial_structure(self):
        """ Reads input structure """
        from .structure_handling import read_structure
        path = self.abspath.join("%s.in" % self.prefix)
        if not path.check(file=True):
            raise IOError("Could not find input file %s" % path)
        return read_structure(str(path))

    def __directory_hook__(self):
        """ Called whenever the directory changes. """
        self.uncache()

    def uncache(self):
        """ Uncache values. """
        self.__dict__.pop("_properties_cache", None)
