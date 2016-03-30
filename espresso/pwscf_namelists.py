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
""" Collections of specialized namelists for Pwscf """
__docformat__ = "restructuredtext en"
__all__ = ['alias', 'Control', 'System', 'Electrons', 'Ions', 'Cell']
from quantities import second, Ry, kilobar
from traitlets import CaselessStrEnum, Unicode, Integer, Bool, Enum, Float
from .namelists import input_transform
from .trait_types import DimensionalTrait, dimensional_trait_as_other
from . import Namelist


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
    prefix = Unicode("pwscf", allow_none=False, help="Prefix for output files")
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
    __set_max_seconds = dimensional_trait_as_other('max_seconds', max_seconds)
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


class System(Namelist):
    """ System namelist """
    nbnd = Integer(default_value=None, allow_none=True, help="Number of bands")
    tot_charge = Float(default_value=None, allow_none=True, help="Total charge of the system")
    tot_magnetization = Float(default_value=None, allow_none=True, help="Total magnetization")
    ecutwfc = DimensionalTrait(Ry, allow_none=True, default_value=None,
                               help="Kinetic energy cutoff for wavefunctions")
    __set_ecutwfc = dimensional_trait_as_other('ecutwfc', ecutwfc)
    ecutrho = DimensionalTrait(Ry, allow_none=True, default_value=None,
                               help="Kinetic energy cutoff for charge density")
    __set_ecutrho = dimensional_trait_as_other('ecutrho', ecutrho)
    ecutfock = DimensionalTrait(Ry, allow_none=True, default_value=None,
                                help="Kinetic energy cutoff for the exact exchange operator")
    __set_ecutfock = dimensional_trait_as_other('ecutfock', ecutfock)
    occupations = Enum(['smearing', 'tetrahedra', 'fixed', 'from_input'], allow_none=True,
                       default_value=None, help="Occupation scheme")
    smearing = Enum(['gaussian', 'gauss', 'methfessel-paxton', 'm-p', 'mp', 'marzari-vanderbilt',
                     'm-z', 'mz', 'fermi-dirac', 'f-d', 'fd'], allow_none=True,
                    default_value=None, help="Smearing function")
    degauss = DimensionalTrait(Ry, allow_none=True, default_value=None,
                               help="Typical energy associated with smearing")
    __set_degauss = dimensional_trait_as_other('degauss', degauss)

    @input_transform
    def __ecutwfc_is_required(self, dictionary, **kwargs):
        from .. import error
        if dictionary.get('ecutwfc', None) is None:
            raise error.ValueError("ecutwfc has not been set. It is a required parameter")


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
    startingpot = CaselessStrEnum(['file', 'atomic'], default_value=None, allow_none=True,
                                  help="Start from existing charge density file\n\n"
                                  "Generally, pylada will handle this value on its own."
                                  "Users are unlikely to need set it themselves.")
    startingwfc = CaselessStrEnum(['atomic', 'atomic+random', 'random', 'file'],
                                  default_value=None, allow_none=True,
                                  help="Start from existing charge density file\n\n"
                                  "When restarting/continuing a calculation, this parameter is "
                                  "automatically set to 'file'.")


class Ions(Namelist):
    """ Ions namelist """
    ion_dynamics = CaselessStrEnum(['bfgs', 'damp', 'verlet', 'langevin', 'langevin-md', 'bfgs',
                                    'beeman'], default_value=None, allow_none=True,
                                   help="Algorithm for ion dynamics")
    dynamics = alias(ion_dynamics)
    ion_positions = CaselessStrEnum(['default', 'from_input'], default_value=None, allow_none=True,
                                    help="When restarting, whether to read from ion positions "
                                    "input or not")
    positions = alias(ion_positions)
    pot_extrapolation = CaselessStrEnum(['none', 'atomic', 'first_order', 'second_order'],
                                        default_value=None, allow_none=True,
                                        help="Extrapolation of the potential from previous "
                                        "ionic steps")
    wfc_extrapolation = CaselessStrEnum(['none', 'first_order', 'second_order'],
                                        default_value=None, allow_none=True,
                                        help="Extrapolation of the wavefunctions from previous "
                                        "ionic steps")
    remove_rigid_rod = Bool(allow_none=True, default_value=None,
                            help="If true, removes spurious rotations during ion dynamics")


class Cell(Namelist):
    """ Cells namelist """
    cell_dynamics = CaselessStrEnum(['none', 'sd', 'damp-pr', 'damp-w', 'bfgs', 'pr', 'w'],
                                    default_value=None, allow_none=True,
                                    help="Algorithm for cell dynamics")
    dynamics = alias(cell_dynamics)
    press = DimensionalTrait(kilobar, allow_none=True, default_value=None, help="External pressure")
    __set_press = dimensional_trait_as_other('press', press)
    press_conv_thr = DimensionalTrait(kilobar, allow_none=True, default_value=None,
                                      help="External pressure convergence threshhold")
    __set_press_conv_thr = dimensional_trait_as_other('press_conv_thr', press_conv_thr)
    cell_factor = Float(allow_none=True, default_value=None, min=0e0,
                        help="Used when constructing pseudopotential tables")
    factor = alias(cell_factor)
    cell_dofree = CaselessStrEnum(['all', 'x', 'y', 'z', 'xy', 'xz', 'yz', 'shape', 'volume',
                                   '2Dxy', '2Dshape'], default_value=None, allow_none=True,
                                  help="Degrees of freedom during relaxation")
    dofree = alias(cell_dofree)
