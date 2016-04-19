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
from traitlets import HasTraits, Instance
from ..tools import stateless, assign_attributes
from . import Namelist
from .card import Card
from .pwscf_namelists import Control, System, Electrons, Ions, Cell, alias
from .namelists import input_transform


class Pwscf(HasTraits):
    """ Wraps up Pwscf in python """
    control = Instance(Control, args=(), kw={}, allow_none=False)
    system = Instance(System, args=(), kw={}, allow_none=False)
    electrons = Instance(Electrons, args=(), kw={}, allow_none=False)
    ions = Instance(Ions, args=(), kw={}, allow_none=False)
    cell = Instance(Cell, args=(), kw={}, allow_none=False)
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
        from .. import error
        from .namelists import InputTransform
        from .misc import write_pwscf_input
        from copy import copy

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

        for transform in self.__class__.__dict__.values():
            if isinstance(transform, InputTransform):
                logger.debug("Transforming input using method %s" % transform.method.__name__)
                transform.method(self, f90namelist, cards=cards, structure=structure, **kwargs)

        return write_pwscf_input(f90namelist, cards, stream)

    @input_transform
    def __add_structure_to_input(self, dictionary=None, cards=None, structure=None, **kwargs):
        from .structure_handling import add_structure
        if structure is None:
            return

        add_structure(structure, dictionary, cards)
        atomic_species = self._write_atomic_species_card(structure)
        # filter cards in-place: we need to modify the input sequence itself
        for i, u in enumerate(list(cards)):
            if u.name in 'atomic_species':
                cards.pop(i)
        cards.append(atomic_species)

    @input_transform
    def __delete_ions_and_cells_if_not_relaxing(self, dictionary, **kwargs):
        if self.control.calculation not in ['relax', 'md', 'vc-relax', 'vc-md']:
            dictionary.pop('ions', None)
        if self.control.calculation not in ['vc-relax', 'vc-md']:
            dictionary.pop('cell', None)

    def read(self, filename, clear=True):
        """ Read from a file """
        from ..misc import local_path
        from .card import read_cards

        # read namelists first
        if clear:
            self.__namelists.clear()
            self.__cards = {}
            for name in self.trait_names():
                value = getattr(self, name)
                if hasattr(value, 'clear'):
                    value.clear()

        filename = local_path(filename)
        logger.info("%s: Reading from file %s", self.__class__.__name__, filename)
        self.__namelists.read(filename)

        traits = set(self.trait_names()).intersection(self.__namelists.names())
        for traitname in traits:
            newtrait = getattr(self.__namelists, traitname)
            delattr(self.__namelists, traitname)
            trait = getattr(self, traitname)
            for key in newtrait.names():
                setattr(trait, key, getattr(newtrait, key))

        # Then read all cards
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
    @assign_attributes(ignore=['overwrite', 'comm', 'restart'])
    def iter(self, structure, outdir=".", comm=None, overwrite=False, restart=None, **kwargs):
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
        from ..misc import local_path
        from .. import pwscf_program
        from ..process.program import ProgramProcess

        outdir = local_path(outdir)
        logger.info('Running Pwscf in: %s' % outdir)
        outdir.ensure(dir=True)

        # check for pre-existing and successful run.
        if not overwrite:
            # Check with this instance's Extract, cos it is this calculation we shall
            # do here. Derived instance's Extract might be checking for other stuff.
            extract = self.Extract(str(outdir))
            if extract.success:
                yield extract  # in which case, returns extraction object.
                return

        # if restarting, gets structure, sets charge density and wavefuntion at start
        # otherwise start passes structure back to caller
        structure = self._restarting(structure, restart, outdir)

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

        stdout = self.control.prefix + ".out"
        stderr = self.control.prefix + ".err"
        stdin = self.control.prefix + ".in"
        yield ProgramProcess(program, cmdline=cmdline, outdir=str(outdir), onfinish=onfinish,
                             stdin=stdin, stdout=stdout, stderr=stderr,
                             dompi=comm is not None)
        # yields final extraction object.
        yield self.Extract(str(outdir))

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

    def _restarting(self, structure, restart, outdir):
        """ Steps to take when restarting

            if restarting, gets structure, sets charge density and wavefuntion at start. Otherwise
            passes structure back to caller.
        """
        from .. import error
        if restart is None:
            return structure

        # normalize: restart could be an Extract object, or a path
        restart = self.Extract(restart)
        if not restart.success:
            logger.critical("Cannot restart from unsuccessful calculation")
            raise error.RuntimeError("Cannot restart from unsuccessful calculation")

        chden = restart.abspath.join('charge-density.xml')
        if chden.check(file=True):
            logger.info("Restarting from charge density %s" % chden)
            chden.copy(outdir.join(chden.basename))
            self.electrons.startingpot = 'file'
        elif self.electrons.startingpot == 'file':
            logger.warning("No charge density found, setting startingpot to atomic")
            self.electrons.startingpot = 'atomic'

        wfcden = restart.abspath.join('%s.wfc1' % self.control.prefix)
        if wfcden.check(file=True):
            logger.info("Restarting from wavefunction file %s" % wfcden)
            wfcden.copy(outdir.join(wfcden.basename))
            self.electrons.startingwfc = 'file'
        elif self.electrons.startingwfc == 'file':
            logger.warning("No wavefunction file found, setting startingwfc to atomic+random")
            self.electrons.startingwfc = 'atomic+random'

        return restart.initial_structure

    def _bring_up(self, structure, outdir, **kwargs):
        """ Prepares for actual run """
        from ..misc import chdir
        logger.info('Preparing directory to run Pwscf: %s ' % outdir)

        with chdir(outdir):
            self.write(structure=structure,
                       stream=outdir.join("%s.in" % self.control.prefix),
                       outdir=outdir, **kwargs)

            self.pseudos_do_exist(structure, verbose=True)
            outdir.join('.pylada_is_running').ensure(file=True)

    def _write_atomic_species_card(self, structure):
        """ Creates atomic-species card """
        from quantities import atomic_mass_unit
        from .. import periodic_table, error
        from .card import Card
        result = Card('atomic_species', value="")
        # Check peudo-files exist
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
        from ..misc import local_path
        directory = local_path(directory)
        if directory.join('.pylada_is_running').check(file=True):
            directory.join('.pylada_is_running').remove()

    @classmethod
    def Extract(class_, outdir, **kwargs):
        from .extract import Extract
        if hasattr(outdir, 'success') and hasattr(outdir, 'directory'):
            return outdir
        return Extract(outdir, **kwargs)

    def __repr__(self):
        from numpy import abs
        from quantities import atomic_mass_unit
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
            print(program)
            raise error.RuntimeError("Pwscf failed to execute correctly.")

        return program
