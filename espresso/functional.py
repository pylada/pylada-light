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
from quantities import bohr_radius, second, Ry
from traitlets import HasTraits, CaselessStrEnum, Unicode, Integer, Instance, Bool, Enum, Float
from ..tools import stateless, assign_attributes
from .namelists import input_transform
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
                                  default_value=None, allow_none=True, help="Task to be performed")
    title = Unicode(None, allow_none=True, help="Title of the calculation")
    verbosity = CaselessStrEnum(['high', 'low'], None, allow_none=True,
                                help="How much talk from Pwscf")
    prefix = Unicode(None, allow_none=True, help="Prefix for output files")
    pseudo_dir = Unicode(None, allow_none=True, help="Directory with pseudo-potential files")
    wf_collect = Bool(allow_none=True, default_value=None,
                      help="If true, saves wavefunctions to readable files")
    nstep = Integer(allow_none=True, default_value=None, min=0,
                    help="Number of ionic + electronic steps")
    iprint = Integer(allow_none=True, default_value=None, min=0,
                     help="Bands are printed every n steps")
    tstress = Bool(allow_none=True, default_value=None, help="Whether to compute stress")
    tprnfor = Bool(allow_none=True, default_value=None, help="Whether to compute forces")
    max_seconds = DimensionalTrait(second, allow_none=True,
                                   default_value=None, help="Jobs stops after n seconds")
    etot_conv_thr = Float(allow_none=True, default_value=None,
                          help="Convergence criteria for total energy")
    force_conv_thr = Float(allow_none=True, default_value=None,
                           help="Convergence criteria for forces")
    disk_io = Enum(['high', 'medium', 'low', 'none'], default_value=None,
                   allow_none=True, help="Amount of disk IO")

    @input_transform
    def _set_outdir(self, dictionary, **kwargs):
        """ Sets output directory from input """
        from os.path import expandvars, expanduser
        if 'outdir' not in kwargs:
            return
        dictionary['outdir'] = expanduser(expandvars(kwargs['outdir']))

    @input_transform
    def _set_max_seconds(self, dictionary, **kwargs):
        value = dictionary.get('max_seconds', None)
        if hasattr(value, 'rescale'):
            dictionary['max_seconds'] = float(value.rescale(second))


class System(Namelist):
    """ System namelist """
    nbnd = Integer(default_value=None, allow_none=True, help="Number of bands")
    tot_charge = Float(default_value=None, allow_none=True, help="Total charge of the system")
    tot_magnetization = Float(default_value=None, allow_none=True, help="Total magnetization")
    ecutwfc = DimensionalTrait(Ry, allow_none=True, default_value=None,
                               help="Kinetic energy cutoff for wavefunctions")
    ecutrho = DimensionalTrait(Ry, allow_none=True, default_value=None,
                               help="Kinetic energy cutoff for charge density")
    ecutfock = DimensionalTrait(Ry, allow_none=True, default_value=None,
                                help="Kinetic energy cutoff for the exact exchange operator")
    occupations = Enum(['smearing', 'tetrahedra', 'fixed', 'from_input'], allow_none=True,
                       default_value=None, help="Occupation scheme")
    smearing = Enum(['gaussian', 'gauss', 'methfessel-paxton', 'm-p', 'mp', 'marzari-vanderbilt',
                     'm-z', 'mz', 'fermi-dirac', 'f-d', 'fd'], allow_none=True,
                    default_value=None, help="Smearing function")
    degauss = DimensionalTrait(Ry, allow_none=True, default_value=None,
                               help="Typical energy associated with smearing")

    @input_transform
    def _set_rydberg_traits(self, dictionary, **kwargs):
        for name in ['degauss', 'ecutfock', 'ecutwfc', 'ecutrho']:
            value = dictionary.get(name, None)
            if hasattr(value, 'rescale'):
                dictionary[name] = float(value.rescale(Ry))


class Electrons(Namelist):
    """ Electrons namelist """
    electron_maxstep = Integer(default_value=None, allow_none=True,
                               help="Maximum number of scf iterations")
    itermax = alias(electron_maxstep)
    conv_thr = Float(allow_none=True, default_value=None,
                     help="Convergence criteria for self consistency")
    mixing_ndim = Integer(allow_none=True, default_value=None, min=0,
                          help="Number of iterations used in mixing")
    mixing_mode = Enum(['plain', 'TF', 'local-TF'], allow_none=True, default_value=None,
                       help="Mixing mode")
    mixing_beta = Float(allow_none=True, default_value=None, help="Mixing factor")
    diagonalization = Enum(['david', 'cg', 'cg-serial'], allow_none=True, default_value=None,
                           help="Diagonalization method")
    diago_cg_max_iter = Integer(allow_none=True, default_value=None, min=0,
                                help="Max number of iterations for CG diagonalization")
    diago_david_ndim = Integer(allow_none=True, default_value=None, min=2,
                               help="Dimension of workspace in David diagonalization")
    diago_full_acc = Bool(allow_none=True, default_value=None,
                          help="Whether to diagonalize empty-states at the same level"
                          "as occupied states")


class Pwscf(HasTraits):
    """ Wraps up Pwscf in python """
    control = Instance(Control, args=(), kw={}, allow_none=False)
    system = Instance(System, args=(), kw={}, allow_none=False)
    electrons = Instance(Electrons, args=(), kw={}, allow_none=False)
    k_points = Instance(Card, args=('K_POINTS',), kw={'subtitle': 'gamma'}, allow_none=False,
                        help="Defines the set of k-points for the calculation")

    kpoints = alias(k_points)

    __private_cards = ['atomic_species']
    """ Cards that are handled differently by Pwscf

        For instance, atomic_species is handled the species attribute.
    """

    def __init__(self, **kwargs):
        from . import Namelist
        super(Pwscf, self).__init__(**kwargs)
        self.__namelists = Namelist()
        self.__cards = {}
        self.species = {}
        """ Dictionary of species that can be used in the calculation

            A specie is an object with at least a 'filename' attribute pointing to the
            pseudo-potential.
        """

    def __getattr__(self, name):
        """ look into extra cards and namelists """
        if name in self.__cards:
            return self.__cards[name]
        elif hasattr(self.__namelists, name):
            return getattr(self.__namelists, name)
        return super(Pwscf, self).__getattribute__(name)

    def add_specie(self, name, pseudo, **kwargs):
        """ Adds a specie to the current known species """
        from .specie import Specie
        self.species[name] = Specie(name, pseudo, **kwargs)

    def write(self, stream=None, structure=None, **kwargs):
        """ Writes Pwscf input

            - if stream is None (default), then returns a string containing namelist in fortran
                format
            - if stream is a string, then it should a path to a file
            - otherwise, stream is assumed to be a stream of some sort, with a `write` method
        """
        from os.path import expanduser, expandvars, abspath
        from .structure_handling import add_structure
        from .. import error
        from .misc import write_pwscf_input
        from copy import copy
        from io import StringIO

        namelist = copy(self.__namelists)
        cards = copy(self.__cards)
        for key in self.trait_names():
            value = getattr(self, key)
            if isinstance(value, Namelist):
                setattr(namelist, key, value)
            elif isinstance(value, Card):
                if value.name in cards:
                    raise error.internal("Found two cards with the same name")
                cards[value.name] = value

        cards = list(cards.values())
        f90namelist = namelist.namelist(structure=structure, **kwargs)
        if structure is not None:
            add_structure(structure, f90namelist, cards)
            atomic_species = self._write_atomic_species_card(structure)
            cards = [u for u in cards if u.name != 'atomic_species']
            cards.append(atomic_species)

        return write_pwscf_input(f90namelist, cards, stream)

    def read(self, filename, clear=True):
        """ Read from a file """
        from os.path import expanduser, expandvars, abspath
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
            elif card.name in self.__private_cards:
                if hasattr(self, '_read_%s_card' % card.name):
                    getattr(self, '_read_%s_card' % card.name)(card)
                else:
                    logger.debug('%s is handled internally' % card.name)
            else:
                self.__cards[card.name] = card

    def add_card(self, name, value=None, subtitle=None):
        """ Adds a new card, or sets the value of an existing one """
        if isinstance(getattr(self, name, None), Card):
            card = getattr(self, name)
        elif card.name in self.__private_cards:
            logger.warn('%s is handled internally' % card.name)
            return
        else:
            logger.info("%s: Adding new card %s", self.__class__.__name__, name)
            card = Card(name)
            self.__cards[name] = card
        card.subtitle = subtitle
        card.value = value

    def add_namelist(self, name, **kwargs):
        """ Adds a new namelist, or sets the value of an existing one """
        from .namelists import Namelist
        if isinstance(getattr(self, name, None), Namelist):
            namelist = getattr(self, name)
            namelist.clear()
        else:
            logger.info("%s: Adding new namelist %s", self.__class__.__name__, name)
            namelist = Namelist()
            setattr(self.__namelists, name, namelist)

        for key, value in kwargs.items():
            setattr(namelist, key, value)

    @stateless
    @assign_attributes(ignore=['overwrite', 'comm'])
    def iter(self, structure, outdir=None, comm=None, overwrite=False, **kwargs):
        """ Allows asynchronous Pwscf calculations

            This is a generator which yields two types of objects:

            .. code:: python

                yield Program(program="Vasp", outdir=outdir)
                yield Extract(outdir=outdir)

            - :py:class:`~pylada.process.program.ProgramProcess`: once started, this process will
               run an actual Pwscf calculation.
            - :py:attr:`Extract`: once the program has been runned, and extraction object is
              yielded, in order that the results of the run can be analyzed.

            :param structure:
                :py:class:`~pylada.crystal.Structure` structure to compute.
            :param outdir:
                Output directory where the results should be stored.  This
                directory will be checked for restart status, eg whether
                calculations already exist. If None, then results are stored in
                current working directory.
            :param comm:
                Holds arguments for executing VASP externally.
            :param overwrite:
                If True, will overwrite pre-existing results. 
                If False, will check whether a successful calculation exists. If
                one does, then does not execute. 
            :param kwargs:
                Any attribute of the Pwscf instance can be overridden for
                the duration of this call by passing it as keyword argument:

                >>> for program in vasp.iter(structure, outdir, sigma=0.5):
                ...

                The above would call VASP_ with smearing of 0.5 eV (unless a
                successfull calculation already exists, in which case the
                calculations are *not* overwritten). 

            :yields: A process and/or an extraction object, as described above.

            :raise RuntimeError: when computations do not complete.
            :raise IOError: when outdir exists but is not a directory.

            .. note::

                This function is stateless. It expects that self and structure can
                be deepcopied correctly.

            .. warning:: 

                This will never overwrite successfull Pwscf calculation, even if the
                parameters to the call are different.
        """
        from .. import pwscf_program
        from ..process.program import ProgramProcess

        logger.info('Running Pwscf in: %s' % outdir)

        # check for pre-existing and successful run.
        if not overwrite:
            # Check with this instance's Extract, cos it is this calculation we shall
            # do here. Derived instance's Extract might be checking for other stuff.
            extract = self.Extract(outdir)
            if extract.success:
                yield extract  # in which case, returns extraction object.
                return

        # copies/creates file environment for calculation.
        self._bring_up(structure, outdir, comm=comm, overwrite=overwrite)

        # figures out what program to call.
        program = getattr(self, 'program', pwscf_program)
        if program == None:
            raise RuntimeError('program was not set in the espresso functional')
        logger.info("Pwscf program: %s" % program)
        cmdline = program.rstrip().lstrip().split()[1:]
        program = program.rstrip().lstrip().split()[0]

        def onfinish(process, error):
            self._bring_down(outdir, structure)

        yield ProgramProcess(program, cmdline=cmdline, outdir=outdir, onfinish=onfinish,
                             stdin='pwscf.in', stdout='stdout', stderr='stderr',
                             dompi=comm is not None)
        # yields final extraction object.
        yield self.Extract(outdir)

    def pseudos_do_exist(self, structure, verbose=False):
        """ True if it all pseudos exist

            :raises error.KeyError: if no species defined
        """
        from .specie import Specie
        from .. import error
        for specie_name in set([u.type for u in structure]):
            if specie_name not in self.species:
                msg = "No specie defined for %s: no way to get pseudopotential" % specie_name
                raise error.KeyError(msg)
            specie = self.species[specie_name]
            if not Specie(specie_name, specie.pseudo).file_exists(self.control.pseudo_dir):
                if verbose:
                    logger.critical(
                        "Specie %s: pseudo = %s" % (specie_name, specie.pseudo))
                return False
        return True

    def _bring_up(self, structure, outdir, **kwargs):
        """ Prepares for actual run """
        from os.path import join
        from ..misc.changedir import Changedir

        logger.info('Preparing directory to run Pwscf: %s ' % outdir)

        with Changedir(outdir) as tmpdir:
            # inputfile = join(tmpdir, "pwscf.in")
            self.write(structure=structure,
                       stream=join(tmpdir, "pwscf.in"), outdir=tmpdir, **kwargs)

            self.pseudos_do_exist(structure, verbose=True)

    def _write_atomic_species_card(self, structure):
        """ Creates atomic-species card """
        from quantities import atomic_mass_unit
        from .. import periodic_table, error
        from .card import Card
        result = Card('atomic_species', value="")
        # Check peudo-files exist
        for specie_name in set([u.type for u in structure]):
            if specie_name not in self.species:
                msg = "No specie defined for %s: no way to get pseudopotential" % specie_name
                raise error.RuntimeError(msg)
            specie = self.species[specie_name]
            mass = getattr(specie, 'mass', None)
            if mass is None:
                mass = getattr(getattr(periodic_table, specie_name, None), 'mass', 1)
            if hasattr(mass, 'rescale'):
                mass = float(mass.rescale(atomic_mass_unit))
            result.value += "%s %s %s\n" % (specie_name, mass, specie.pseudo)
        return result

    def _read_atomic_species_card(self, card):
        """ Adds atomic specie info to species dictionary """
        for line in card.value.rstrip().lstrip().split('\n'):
            name, mass, pseudo = line.split()
            if name in self.species:
                self.species[name].pseudo = pseudo
                self.species[name].mass = float(mass)
            else:
                self.add_specie(name, pseudo, mass=float(mass))

    def _bring_down(self, directory, structure):
        from os.path import exists
        from os import remove
        from . import files
        from ..misc import Changedir

        with Changedir(directory) as pwd:
            if exists('.pylada_is_running'):
                remove('.pylada_is_running')

    @classmethod
    def Extract(class_, outdir):
        from collections import namedtuple
        return namedtuple('Extract', ['success'])(False)

    def __repr__(self):
        from numpy import abs
        from quantities import atomic_mass_unit
        from itertools import chain
        result = "pwscf = %s()\n" % self.__class__.__name__
        for k, v in self.__dict__.items():
            if k[0] != '_' and k != 'species':
                result += "pwscf.%s = %s\n" % (k, repr(v))

        for name, value in self.species.items():
            result += "pwscf.add_specie('%s', '%s'" % (name, value.pseudo)
            if abs(value.mass - atomic_mass_unit) > 1e-12:
                result += ", mass=%s" % float(value.mass.rescale(atomic_mass_unit))
            for k, v in value.__dict__.items():
                if k[0] != '_' and k not in ['name', 'pseudo']:
                    result += ", %s=%s" % (k, repr(v))
            result += ")\n"

        for k, v in self.__cards.items():
            if k[0] != '_':
                result += "pwscf.%s = %s\n" % (k, repr(v))

        for name in self.__namelists.names():
            result += "pwscf.add_namelist(%s" % name
            value = getattr(self, name)
            for k in value.names():
                v = getattr(value, k)
                result += ", %s=%s" % (k, repr(v))
            result += ")"

        for name in self.trait_names():
            value = getattr(self, name)
            if hasattr(value, 'printattr'):
                result += value.printattr("pwscf." + name)

        return result

    def __call__(self, structure, outdir=None, comm=None, overwrite=False, **kwargs):
        """ Blocking call to pwscf

            :returns: An extraction object of type :py:attr:`Extract`.
        """
        from .. import error
        for program in self.iter(
                structure, outdir=outdir, comm=comm, overwrite=overwrite, **kwargs):
            # iterator may yield the result from a prior successful run.
            if getattr(program, 'success', False):
                continue
            # If the following is True, then the program failed to run correctly.
            if not hasattr(program, 'start'):
                break
            # otherwise, it should yield a Program process to execute.
            # This next line starts the asynchronous call to the external VASP
            # program.
            program.start(comm)

            # This next lines waits until the VASP program is finished.
            program.wait()

        # Last yield should be an extraction object.
        if not program.success:
            raise error.RuntimeError("Pwscf failed to execute correctly.")

        return program
