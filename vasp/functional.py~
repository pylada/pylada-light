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

# -*- coding: utf-8 -*-
""" Sub-package containing the functional. """
__docformat__ = "restructuredtext en"
__all__ = ['Vasp']
from ..tools import stateless, assign_attributes
from ..tools.input import AttrBlock
from ..misc import add_setter
from .extract import Extract as ExtractVasp
from pylada.misc import bugLev
from pylada.misc import testValidProgram


class Vasp(AttrBlock):
  """ Interface to VASP code.
  
      This interface makes it a bit easier to call VASP_ both for
      high-throughput calculations and for complex calculations involving more
      that one actual call to the program.

      The main way to use the code is as follows.

      First, one creates the actual functional:

      >>> from pylada.vasp import Vasp
      >>> functional = Vasp()

      The one sets it up, taking care that all the required pseudo-potentials
      are defined:

      >>> vasp.add_specie = 'H', '$HOME/pseudos/PAWGGA/H'
      >>> vasp.add_specie = 'C', '$HOME/pseudos/PAWGGA/C'
      >>> vasp.sigma = 0.2
      >>> vasp.prec = 'accurate'
      >>> vasp.lcharg = True

      The parameters of the functional have the same name as the tags in the
      INCAR_. The syntax is also the same, except that integers, floats, and
      booleans can be used wherever it makes sense.

      Although a fair number of parameters already exist, some are missing. It
      is possible to add them with:

      >>> vasp.add_keyword('tagname', value)

      The tagname needs not be given in capital letters. The value can be a
      string, in which case it will be printed in the INCAR_ as is, basic
      types, such as integers, floats, and booleans, or list of such types. It
      is then possible to access the new parameter as with any other:

      >>> vasp.tagname = 5
      >>> vasp.tagname
      5

      There are few special parameters with enhanced behaviors, such as
      :py:attr:`magmom`, :py:attr:`encut`, or :py:attr:`ediff_per_atom`. 
      Any parameter can be completely disabled with:

      >>> vasp.sigma = None

      In this case, SIGMA_ will simply not appear in the INCAR_. 

      The functional is then called with two or more arguments:

      >>> result = vasp(structure, 'this/directory', sigma=-1)

      The first argument is a :py:class:`pylada.crystal.Structure` instance on
      which to perform the calculation. The species in the structure should all
      have a pseudo-potential declared vias :py:attr:`add_specie`. The second
      argument is the directory where the calculations should take place. Other
      arguments, such as ``sigma`` above will change the appropriate parameter
      for the duration of the call (the call is stateless). 

      The result is an extraction object capable of grepping different
      properties from the OUTCAR. The major property is
      :py:attr:`~pylada.vasp.extract.Extract.success`. For more, please see
      :py:class:`~pylada.vasp.extract.Extract`.

      The best way to use this functional is in conjunction with the
      high-throughput interface :py:mod:`pylada.jobfolder`.
  """
  Extract = staticmethod(ExtractVasp)
  """ Extraction class.
  
      This extraction class is used to grep output from an OUTCAR file.
      This attribute merely describes the type of the extraction object as
      :py:class:`~pylada.vasp.extract.Extract`.
  """

  def __init__(self, copy=None, species=None, kpoints=None, **kwargs):
    """ Initializes vasp class. """
    from .keywords import BoolKeyword, Magmom, System, Npar, ExtraElectron,    \
                          NElect, Algo, Ediff, Ediffg, Encut, EncutGW, IStart, \
                          ICharg, IStruc, LDAU, PrecFock, Precision, Nsw,      \
                          Isif, IBrion, Relaxation, ISmear, LSorbit, Sigma,    \
                          LMaxMix, LVHar, EdiffPerAtom, EdiffgPerAtom, NonScf
    from ..tools.input import TypedKeyword, ChoiceKeyword
    super(Vasp, self).__init__()
    if bugLev >= 5:
      print "vasp/functional.Vasp.__init__: species: ", species
      print "vasp/functional.Vasp.__init__: kpoints: ", kpoints
      print "vasp/functional.Vasp.__init__: kwargs: ", kwargs

    self.species = species if species is not None else {}
    """ Species in the system.
    
        Defines both POTCAR_, U, and/or NLEP parameters.  This is generally
        done using the method :py:meth:`add_specie`.

        .. _POTCAR: http://cms.mpi.univie.ac.at/wiki/index.php/POTCAR
    """
    self.kpoints = kpoints if kpoints is not None \
                   else "\n0\nM\n4 4 4\n0 0 0"
    """ kpoints for which to perform calculations. """
    self.restart = kwargs.get('restart', None)
    """ Calculation from which to restart. 

        Depending on the values of :py:attr:`istart`, :py:attr:`icharg`, and
        :py:attr:`istruc`, this calculation will copy the charge density,
        wavefunctions, and structure from this object. It should be either
        None, or an extraction object returned by a previous calculation (e.g.
        :py:class:`~pylada.vasp.extract.Extract`):

        .. code-block:: python

           calc1 = vasp(structure)
           calc2 = vasp(structure, restart=calc2, nonscf=True)

        The snippet above performs a non-self-consistent calculation using the
        first calculation. In this example, it is expected that
        :py:attr:`istart`, :py:attr:`icharg`, and :py:attr:`istruc` are all set
        to 'auto', in which case Pylada knows to do the right thing, e.g. copy
        whatever is available, and nothing is ``vasp.restart is None``.

        .. note:: 
        
           The calculation from which to restart needs be successful, otherwise
           it is not considered.

        .. seealso:: :py:attr:`istart`, :py:attr:`istruc`, :py:attr:`icharg`
    """

    self.program = kwargs.get('program', None)
    """ Path to vasp program. 
    
        Can be one of the following:

          - None: defaults to :py:attr:`~pylada.vasp_program`.
            :py:attr:`~pylada.vasp_program` can take the same values as described
            here, except for None.
          - string: Should be the path to the vasp executable. It can be either
            a full path, or an executable within the environment's $PATH
            variable.
          - callable: The callable is called with a :py:class:`Vasp` as sole
            argument. It should return a string, as described above.  In other
            words, different vasp executables can be used depending on the
            parameters. 
    """
    self.addgrid = BoolKeyword()
    """ Adds additional support grid for augmentation charge evaluation. 

        Can be only True or False (or None for VASP_ default).

        .. seealso:: ADDGRID_
        .. _ADDGRID: http://cms.mpi.univie.ac.at/wiki/index.php/ADDGRID
    """
    self.ispin   = ChoiceKeyword(values=(1, 2))
    """ Whether to perform spin-polarized or spin-unpolarized calculations.

        Can be only 1 or 2 (or None for VASP_ default).

        .. seealso:: ISPIN_ 
        .. _ISPIN: http://cms.mpi.univie.ac.at/wiki/index.php/ISPIN
    """
    self.istart    = IStart(value='auto')
    """ Starting wavefunctions.
    
        This tag is about which wavefunction (WAVECAR_) file to read from, if
        any.  It is best to keep this attribute set to -1, in which case, Pylada
        takes care of copying the relevant files.
    
          - -1, 'auto': (Default) Automatically determined by Pylada. Depends on
            the value of :py:attr:`restart` and the existence of the relevant
            files. If a WAVECAR_ file exists, then ISTART_ will be set to 1
            (constant cutoff).
    
          - 0, 'scratch': Start from scratch.
    
          - 1, 'cutoff': Restart with constant cutoff.
    
          - 2, 'basis': Restart with constant basis.
    
          - 3, 'full': Full restart, including TMPCAR.

        This attribute can be set equivalently using an integer or a string, as
        shown above. In practice, the integers will be converted to strings
        within the python interface:

          >>> vasp.istart = 0
          >>> vasp.istart
          'scratch'

        .. note::
        
           Files are copied right before the calculation takes place, not when
           the attribute is set.
    
        .. seealso::
           
           ISTART_, :py:attr:`icharg`, :py:attr:`istruc`, :py:attr:`restart`

        .. _ISTART: http://cms.mpi.univie.ac.at/wiki/index.php/ISTART
    """ 
    self.icharg    = ICharg('auto')
    """ Charge from which to start. 
    
        This tag decides whether to restart from a previously calculated charge
        density, or not. It is best to keep this attribute set to -1, in which
        case, Pylada takes care of copying the relevant files.
    
          - -1: (Default) Automatically determined by Pylada. Depends on the
                value of :py:attr:`restart` and the existence of the relevant
                files. Also takes care of non-scf bit.
    
          - 0: Tries to restart from wavefunctions. Uses the latest WAVECAR_
               file between the one currently in the output directory and the
               one in the restart directory (if specified). Sets
               :py:attr:`nonscf` to False.
    
               .. note:: CHGCAR_ is also copied, just in case.
    
          - 1: Tries to restart from wavefunctions. Uses the latest WAVECAR_
               file between the one currently in the output directory and the
               one in the restart directory (if specified). Sets
               :py:attr:`nonscf` to False.
    
          - 2: Superimposition of atomic charge densities. Sets
               :py:attr:`nonscf` to False.
    
          - 4: Reads potential from POT file (VASP-5.1 only). The POT file is
               deduced the same way as for CHGAR and WAVECAR_ above. Sets
               :py:attr:`nonscf` to False.
    
          - 10, 11, 12: Same as 0, 1, 2 above, but also sets :py:attr:`nonscf`
               to True.  This is a shortcut. The value is actually kept to 0,
               1, or 2:
    
               >>> vasp.icharg = 10
               >>> vasp.nonscf, vasp.icharg
               (True, 0)
    
        .. note::
        
           Files are copied right before the calculation takes place, not
           before.
    
        .. seealso::
        
           ICHARG_, :py:attr:`nonscf`, :py:attr:`restart`, :py:attr:`istruc`,
           :py:attr:`istart`
    
        .. _ICHARG: http://cms.mpi.univie.ac.at/wiki/index.php/ICHARG
    """ 
    self.istruc    = IStruc('auto')
    """ Initial structure. 
    
        Determines which structure is written to the POSCAR. In practice, it
        makes it possible to restart a crashed job from the latest contcar.
        There are two possible options:
    
          - auto: Pylada determines automatically what to use. If a CONTCAR
                  exists in either the current directory or in the restart
                  directory (if any), then uses the latest. Otherwise, uses
                  input structure.
          - scratch: Always uses input structure.
          - contcar: Always use CONTCAR_ structure unless ``overwrite ==
            True``.
          - input: Always use input structure, never :py:attr:`restart` or
            CONTCAR_ structure.
    
        If the run was given the ``overwrite`` option, then always uses the
        input structure.
    
        .. note:: There is no VASP equivalent to this option.
        .. seealso:: :py:attr:`restart`, :py:attr:`icharg`, :py:attr:`istart`
    
        .. _CONTCAR: http://cms.mpi.univie.ac.at/vasp/guide/node60.html
    """
    self.isym      = ChoiceKeyword(values=range(-1, 3))
    """ Symmetry scheme.

        .. seealso:: ISYM_
        .. _ISYM: http://cms.mpi.univie.ac.at/vasp/guide/node115.html
    """ 

    self.lmaxmix   = LMaxMix()
    """ Cutoff *l*-quantum number of PAW charge densities passed to mixer 

        .. seealso:: LMAXMIX_ 
        .. _LMAXMIX: http://cms.mpi.univie.ac.at/wiki/index.php/LMAXMIX
    """

    self.lvhar     = LVHar()

    self.lorbit    = ChoiceKeyword(values=(0, 1, 2, 5, 10, 11, 12))
    """ Decides whether PROOUT and PROOCAR are writtent to disk.

        Can be one of 0|1|2|5|10|11|12|None. 

        .. seealso:: LORBIT_ 
        .. _LORBIT: http://cms.mpi.univie.ac.at/wiki/index.php/LORBIT
    """
    self.nbands    = TypedKeyword(type=int)
    """ Number of bands in the calculation.

        Can be any integer.

        .. seealso:: NBANDS_
        .. _NBANDS: http://cms.mpi.univie.ac.at/wiki/index.php/NBANDS
    """
    self.nomega    = TypedKeyword(type=int)
    """ Number of frequency grid points. 

        Can be any integer.

        .. seealso:: NOMEGA_
        .. _NOMEGA: http://cms.mpi.univie.ac.at/wiki/index.php/NOMEGA
    """
    self.nupdown   = TypedKeyword(type=int)
    self.symprec   = TypedKeyword(type=float)
    self.lwave     = BoolKeyword(value=False)
    """ Whether or not to write the wavefunctions to the WAVECAR_.

        Must be a boolean. Defaults to False (unlike VASP_).

        .. seealso:: LWAVE_
        .. _LWAVE: http://cms.mpi.univie.ac.at/wiki/index.php/LWAVE
        .. _WAVECAR: http://cms.mpi.univie.ac.at/wiki/index.php/WAVECAR
    """
    self.lcharg    = BoolKeyword(value=True)
    """ Whether or not to write the charge density to the CHGCAR_.

        Must be a boolean. Defaults to True.

        .. seealso:: LWAVE_
        .. _LWAVE: http://cms.mpi.univie.ac.at/wiki/index.php/LWAVE
        .. _CHGCAR: http://cms.mpi.univie.ac.at/wiki/index.php/CHGCAR
    """
    self.lvtot     = BoolKeyword(value=False)
    """ Whether or not to write the local potential to the LOCPOT file. 

        Must be a boolean. Defaults to False.

        .. seealso:: LVTOT_
        .. _LVTOT: http://cms.mpi.univie.ac.at/wiki/index.php/LVTOT
    """
    self.lrpa      = BoolKeyword()
    """ Whether to include local field effects at the Hartree level only.

        Must be a boolean or None (leaves default to VASP_).

        .. seealso:: LRPA_
        .. _LRPA: http://cms.mpi.univie.ac.at/wiki/index.php/LRPA
    """ 
    self.loptics   = BoolKeyword()
    """ Whether to compute the frequency dependent dielectic matrix.

        Must be a boolean or None (leaves default to VASP_).

        .. seealso:: LOPTICS_
        .. _LOPTICS: http://cms.mpi.univie.ac.at/wiki/index.php/LOPTICS
    """
    self.lpead     = BoolKeyword()
    """ Compute the finite-difference k-derivative of the wavefunctions.

        Must be a boolean or None (leaves default to VASP_).

        .. seealso:: LPEAD_
        .. _LPEAD: http://cms.mpi.univie.ac.at/wiki/index.php/LPEAD
    """


    self.lplane     = BoolKeyword()


    self.nelm      = TypedKeyword(type=int)
    """ Maximum number of self-consistent electronic minimization steps.
       
        Must be an integer or None (leaves default to VASP_).

        .. seealso:: NELM_
        .. _NELM: http://cms.mpi.univie.ac.at/wiki/index.php/NELM
    """
    self.nelmin    = TypedKeyword(type=int)
    """ Minimum number of self-consistent electronic minimization steps.
       
        Must be an integer or None (leaves default to VASP_).

        .. seealso:: NELMIN_
        .. _NELMIN: http://cms.mpi.univie.ac.at/wiki/index.php/NELMIN
    """
    self.nelmdl    = TypedKeyword(type=int)
    """ Number of non-selfconsistent steps at the beginning. 
       
        Must be an integer or None (leaves default to VASP_).

        .. seealso:: NELMDL_
        .. _NELMDL: http://cms.mpi.univie.ac.at/wiki/index.php/NELMDL
    """
    self.ngx       = TypedKeyword(type=int)
    """ number of grid points in the FFT-grid along the first lattice vector.

        Must be an integer or None (leaves default to VASP_).

        .. seealso:: NGX_, :py:attr:`ngy`, :py:attr:`ngz`
        .. _NGX: http://cms.mpi.univie.ac.at/wiki/index.php/NGX
    """
    self.ngy       = TypedKeyword(type=int)
    """ number of grid points in the FFT-grid along the second lattice vector.

        Must be an integer or None (leaves default to VASP_).

        .. seealso:: NGY_, :py:attr:`ngx`, :py:attr:`ngz`
        .. _NGY: http://cms.mpi.univie.ac.at/wiki/index.php/NGY
    """
    self.ngz       = TypedKeyword(type=int)
    """ number of grid points in the FFT-grid along the third lattice vector.

        Must be an integer or None (leaves default to VASP_).

        .. seealso:: NGZ_, :py:attr:`ngx`, :py:attr:`ngy`
        .. _NGZ: http://cms.mpi.univie.ac.at/wiki/index.php/NGZ
    """
    self.nonscf    = NonScf()
    """ If True, performs a non-self consistent calculation.

        The value of this keyword is checked by :py:attr:`icharg` and used
        appropriately. The attribute :py:attr:`lsorbit` also acts and checks
        on it. It is False by default.
    """
    
    self.magmom    = Magmom()
    """ Sets the initial magnetic moments on each atom.
    
        There are three types of usage: 
    
        - if None or False, does nothing
        - if calculations are not spin-polarized, does nothing.
        - if a string, uses that as for the MAGMOM_ keyword
        - if True and at least one atom in the structure has a non-zero
          ``magmom`` attribute, then creates the relevant moment input for
          VASP_
    
        If the calculation is **not** spin-polarized, then the magnetic moment
        tag is not set.
    
        .. note:: Please set by hand for non-collinear calculations
        .. seealso:: MAGMOM_
        .. _MAGMOM: http://cms.mpi.univie.ac.at/wiki/index.php/MAGMOM
    """
    self.system    = System()
    """ System title to use for calculation.
    
        - If None and ... 
           - if the structure has a ``name`` attribute, uses that as the
             calculations title
           - else does not use SYSTEM_ tag
        - If something else which is convertible to a string,  and ...
           - if the structure has a ``name`` attribute, uses 'string: name' as
             the title
           - otherwise, uses the string
    
        .. seealso:: SYSTEM_
        .. _SYSTEM: http://cms.mpi.univie.ac.at/vasp/guide/node94.html>
    """
    self.npar      = Npar()
    """ Parallelization over bands. 
    
        Npar defines how many nodes work on one band.
        It can be set to a particular number:
    
        >>> vasp.npar = 2
    
        Or it can be deduced automatically. Different schemes are available:
        
          - power of two: npar is set to the largest power of 2 which divides
            the number of processors.
   
            >>> vasp.npar = "power of two"
    
            If the number of processors is not a power of two, prints nothing.
    
          - square root: npar is set to the square root of the number of
            processors.
    
            >>> vasp.npar = "sqrt"
        
    
        .. seealso: NPAR_ 
        .. _NPAR: http://cms.mpi.univie.ac.at/vasp/guide/node138.html>
    """
    self.extraelectron = ExtraElectron()
    """ Number of electrons relative to neutral system.
        
        Gets the number of electrons in the (neutral) system. Then adds value to
        it and computes with the resulting number of electrons.
    
        >>> vasp.extraelectron =  0  # charge neutral system
        >>> vasp.extraelectron =  1  # charge -1 (1 extra electron)
        >>> vasp.extraelectron = -1  # charge +1 (1 extra hole)

        Disables :py:attr:`nelect` if set to anything but None: these two
        properties are mutually exclusive.
    
        .. seealso:: NELECT_, :py:attr:`nelect`
        .. _NELECT: http://cms.mpi.univie.ac.at/wiki/index.php/NELECT
    """
    self.nelect = NElect()
    """ Sets the absolute number of electrons.
        
        Disables :py:attr:`extraelectron` if set to something other than None.
    
        .. seealso:: NELECT_, :py:attr:`extraelectron`
    """
    self.algo = Algo()
    """ Electronic minimization. 
    
        Defines the kind of algorithm vasp will run.
          - very fast
          - fast, f (default)
          - normal, n
          - all, a
          - damped, d 
          - Diag 
          - conjugate, c (vasp 5)
          - subrot (vasp 5)
          - eigenval (vasp 5)
          - Nothing (vasp 5)
          - Exact  (vasp 5)
          - chi
          - gw
          - gw0
          - scgw
          - scgw0
    
        If :py:data:`~pylada.is_vasp_4` is an existing configuration variable of
        :py:mod:`pylada` the parameters marked as vasp 5 will fail.
    
        .. warning:: The string None is not  allowed, as it would lead to
           confusion with the python object None. Please use "Nothing" instead.
           The python object None will simply not print the ALGO_ keyword to
           the INCAR_ file.
    
        .. note:: By special request, "fast" is the default algorithm.
    
        .. seealso:: ALGO_
        .. _ALGO: http://cms.mpi.univie.ac.at/vasp/vasp/ALGO_tag.html
    """ 
    self.ediff = Ediff()
    """ Sets the absolute energy convergence criteria for electronic minimization.
    
        EDIFF_ is set to this value in the INCAR_. 
    
        Disables :py:attr:`ediff_per_atom` if set to anything but None. These
        two properties are mutually exclusive. If negative or null, defaults to
        zero.
    
        .. seealso:: EDIFF_, :py:attr:`ediff_per_atom` 

        .. _EDIFF: http://cms.mpi.univie.ac.at/wiki/index.php/EDIFF
    """
    self.ediff_per_atom = EdiffPerAtom()
    """ Sets the relative energy convergence criteria for electronic minimization.
    
        EDIFF_ is set to this value *times* the number of atoms in the
        structure.  This approach is more sensible than straight-off
        :py:attr:`ediff` when doing high-throughput over many structures.
        
        Disables :py:attr:`ediff` if set to anything but None. These two
        properties are mutually exclusive. If negative or null, defaults to
        zero.
    
        .. seealso:: EDIFF_, :py:attr:`ediff`
    """
    self.ediffg = Ediffg()
    """ Sets the absolute energy convergence criteria for ionic relaxation.
    
        EDIFFG_ is set to this value in the INCAR_. 
    
        Disables :py:attr:`ediffg_per_atom` if set to anything but None. These
        two properties are mutually exclusive.
    
        .. seealso:: EDIFFG_, :py:attr:`ediffg_per_atom`
        .. _EDIFFG: http://cms.mpi.univie.ac.at/wiki/index.php/EDIFFG
    """
    self.ediffg_per_atom = EdiffgPerAtom()
    """ Sets the relative energy convergence criteria for ionic relaxation.
  
        - if positive: EDIFFG_ is set to this value *times* the number of atoms
          in the structure. This means that the criteria is for the total energy per atom.
        - if negative: same as a negative EDIFFG_, since that convergence
          criteria is already per atom.
        
        This approach is more sensible than straight-off :py:attr:`ediffg` when
        doing high-throughput over many structures.  Disables :py:attr:`ediffg`
        if set to anything but None. These two properties are mutually
        exclusive.
  
        .. seealso:: EDIFFG_, :py:attr:`ediffg`
    """
    self.encut = Encut()
    """ Defines cutoff factor for calculation. 
    
        There are three ways to set this parameter:
    
        - if value is floating point and 0 < value <= 3: then the cutoff is
          ``value * ENMAX``, where ENMAX is the maximum recommended cutoff for
          the species in the system.
        - if value > 3 eV, then ENCUT_ is exactly value (in eV). Any energy
          unit is acceptable.
        - if value < 0 eV or None, does not print anything to INCAR_. 
        
        .. seealso:: ENCUT_
        .. _ENCUT: http://cms.mpi.univie.ac.at/wiki/index.php/ENCUT
    """
    self.encutgw = EncutGW()
    """ Defines cutoff factor for GW calculation. 
  
        There are three ways to set this parameter:
  
        - if value is floating point and 0 < value <= 3: then the cutoff is
          ``value * ENMAX``, where ENMAX is the maximum recommended cutoff for
          the species in the system.
        - if value > 3 eV, then ENCUTGW_ is exactly value (in eV). Any energy
          unit is acceptable.
        - if value < 0 eV or None, does not print anything to INCAR_. 
        
        .. seealso:: ENCUTGW_
        .. _ENCUTGW: http://cms.mpi.univie.ac.at/wiki/index.php/GW_calculations
    """
    self.ldau = LDAU()
    """ Sets U, nlep, and enlep parameters. 
   
        The U, nlep, and enlep parameters of the atomic species are set at the
        same time as the pseudo-potentials. This object merely sets up the INCAR_
        with right input.
    
        However, it does accept one parameter, which can be "off", "on", "occ" or
        "all", and defines the level of verbosity of VASP (with respect to U and nlep).
    
        .. seealso:: LDAU_, LDAUTYPE_, LDAUL_, LDAUJ_, :py:attr:`species`

        .. _LDAU: http://cms.mpi.univie.ac.at/wiki/index.php/LDAU
        .. _LDAUTYPE: http://cms.mpi.univie.ac.at/wiki/index.php/LDAUTYPE
        .. _LDAUL: http://cms.mpi.univie.ac.at/wiki/index.php/LDAUL
        .. _LDAUU: http://cms.mpi.univie.ac.at/wiki/index.php/LDAUU
        .. _LDAUJ: http://cms.mpi.univie.ac.at/wiki/index.php/LDAUJ
    """
    self.precfock = PrecFock()
    """ Sets up FFT grid in hartree-fock related routines.
        
        Allowable options are:
    
        - low
        - medium
        - fast
        - normal
        - accurate
    
        .. seealso:: PRECFOCK_
        .. _PRECFOCK: http://cms.mpi.univie.ac.at/wiki/index.php/PRECFOCK
    """
    self.prec = Precision()
    """ Sets accuracy of calculation. 
    
        - accurate (default)
        - low
        - medium
        - high
        - single
    
        .. seealso:: PREC_
        .. _PREC: http://cms.mpi.univie.ac.at/wiki/index.php/PREC
    """
    self.nsw = Nsw()
    """ Maxium number of ionic iterations. 

        .. seealso:: NSW_
        .. _NSW: http://cms.mpi.univie.ac.at/wiki/index.php/NSW
    """
    self.ibrion = IBrion()
    """ Ions/cell-shape/volume optimization method.
    
        Can only take a restricted set of values: -1 | 0 | 1 | 2 | 3 | 5 | 6 | 7 | 8 | 44.

        .. seealso::
        
           IBRION_, :py:attr:`relaxation`, :py:attr:`isif`, :py:attr:`nsw`

        .. _IBRION: cms.mpi.univie.ac.at/wiki/index.php/IBRIONN
    """
    self.isif = Isif()
    """ Degree of librerty to optimize during geometry optimization

        .. seealso:: 
        
           ISIF_, :py:attr:`relaxation`, :py:attr:`ibrion`, :py:attr:`nsw`

        .. _ISIF: http://cms.mpi.univie.ac.at/vasp/guide/node112.html
    """
    self.relaxation = Relaxation()
    """ Short-cut for setting up relaxation. 

        It accepts two parameters:
        
          - static: for calculation without geometric relaxation.
          - combination of ionic, volume, cellshape: for the type of relaxation
            requested.

        It makes sure that :py:attr:`isif`, :py:attr:`ibrion`, and
        :py:attr:`nsw` take the right value for the
        kind of relaxation.

        .. seealso:: :py:attr:`isif`, :py:attr:`ibrion`, :py:attr:`nsw`
    """
    self.ismear = ISmear()
    """ Smearing function 

        Vasp allows a number of options:

        - metal (-5): Tetrahedron method with Blochl correction (requires a
          |Gamma|-centered *k*-mesh)
        - tetra (-4): Tetrahedron method (requires a |Gamma|-centered *k*-mesh)
        - dynamic (-3): Performs a loop over smearing parameters supplied in
          :py:attr:`smearings`
        - fixed: (-2): Fixed occupation, set *via* :py:attr:`ferwe` and
          :py:attr:`ferdo`
        - fermi (-1): Fermi function
        - gaussian (0): Gaussian function
        - mp n (n>0): Methfessel-Paxton smearing function of order n

        .. seealso::
        
          ISMEAR_, :py:attr:`sigma`, :py:attr:`smearings`, :py:attr:`ferwe`,
          :py:attr:`ferdo`
        
        .. _ISMEAR: http://cms.mpi.univie.ac.at/wiki/index.php/ISMEAR
        .. |Gamma|  unicode:: U+00393 .. GREEK CAPITAL LETTER GAMMA
    """
    self.sigma = Sigma()
    """ Width of the smearing function.

        Accepts floating points which may be signed with an energy unit using
        the quantities_ package:

          .. code-block:: python

            from quantities import eV, hartree

            vasp.smearing = 0.2           # Defaults to eV.
            vasp.smearing = 0.2 * eV      # Same as above, but more explicit.
            vasp.smearing = 0.5 * hartree # if you are so inclined.
    
        .. seealso:: SIGMA_, quantities_, :py:attr:`ismear`
        .. _SIGMA: http://cms.mpi.univie.ac.at/wiki/index.php/SIGMA
        .. _quantities: http://packages.python.org/quantities/index.html
    """
    self.smearings = TypedKeyword(type=[float])
    """ Smearing steps for ``ISMEAR=-3``.

        List of floating points. Does not accept quantities_.

        .. seealso:: ISMEAR_, :py:attr:`ismear`
    """
    self.ferwe = TypedKeyword(type=[float])
    """ Occupancy of the states for ``ISMEAR=-2``.

        List of floating points.
         
        .. seealso:: FERWE_, :py:attr:`ismear`, :py:attr:`ferdo`
        .. _FERWE: http://cms.mpi.univie.ac.at/wiki/index.php/FERWE
    """
    self.ferdo = TypedKeyword(type=[float])
    """ Occupancy of the down-spin states for ``ISMEAR=-2``.

        List of floating points.
         
        .. seealso:: FERDO_, :py:attr:`ismear`, :py:attr:`ferwe`
        .. _FERDO: http://cms.mpi.univie.ac.at/wiki/index.php/FERDO
    """
    self.lsorbit = LSorbit()
    """ Run calculation with spin-orbit coupling. 
    
        Accepts None, True, or False. If True, then sets :py:attr:`nonscf` to
        True and :py:attr:`ispin` to 2.
    """ 
    
    # copies values from other functional.
    if copy is not None: 
      self._input.update(copy._input)
      for key, value in copy.__dict__.iteritems():
        if key in kwargs: continue
        elif key == '_input': continue
        elif hasattr(self, key): setattr(self, key, value)


    # sets all known keywords as attributes.
    for key, value in kwargs.iteritems():
      if hasattr(self, key): setattr(self, key, value)

  # NEVER CALLED!
  def __call__( self, structure, outdir=None, comm=None, overwrite=False, 
                **kwargs):
    """ Calls vasp program and waits until completion. 
    
        This function makes a blocking call to the VASP_ external program. It
        will return only once the calculation is complete. To make asynchronous
        calls to VASP_, please consider using :py:meth:`iter`.

        For a description of the parameters, please see :py:meth:`iter`.

        :returns: An extraction object of type :py:attr:`Extract`.
    """
    import os, sys, traceback

    # NEVER CALLED!
    #print 'functional __call__ A: ===== start stack trace'
    #traceback.print_stack( file=sys.stdout)
    #print 'functional __call__ A: ===== end stack trace'

    if bugLev >= 5:
      print "vasp/functional __call__: outdir: ", outdir
      print "vasp/functional __call__: structure:\n", structure
      print "vasp/functional __call__: comm: ", comm

    for program in self.iter(structure, outdir=outdir, comm=comm, overwrite=overwrite, **kwargs):
      # iterator may yield the result from a prior successful run. 
      if getattr(program, 'success', False): continue
      # If the following is True, then the program failed to run correctly.
      if not hasattr(program, 'start'): break
      # otherwise, it should yield a Program process to execute.
      # This next line starts the asynchronous call to the external VASP
      # program.

      if bugLev >= 5: print "vasp/functional __call__: before start"
      program.start(comm)

      if bugLev >= 5: print "vasp/functional __call__: before wait"
      # This next lines waits until the VASP program is finished.
      program.wait()

      if bugLev >= 5: print "vasp/functional __call__: after wait"
    # Last yield should be an extraction object.
    if not program.success:
      raise RuntimeError("Vasp failed to execute correctly.")
    return program

  @stateless
  @assign_attributes(ignore=['overwrite', 'comm'])
  def iter(self, structure, outdir=None, comm=None, overwrite=False, **kwargs):
    """ Allows asynchronous vasp calculations
     
        This is a generator which yields two types of objects: 

           - :py:class:`~pylada.process.program.ProgramProcess`: once started,
             this process will run an actual VASP_ calculation.
           - :py:attr:`Extract`: once the program has been runned, and
             extraction object is yielded, in order that the results of the run
             can be analyzed. 

        In a new calculation, an instance of each type will be yielded, in the
        order of their description above. It is expected that program is runned
        first, using the first object, before looping to the second object. 
        Thie generator function makes it possible to run different instances of
        VASP_ simultaneously. It also makes it possible to create more complex
        calculations which necessitate more than one actual call to VASP_ (see
        :py:class:`~pylada.vasp.relax.iter_relax`), while retaining the ability
        to run multiple VASP_ programs simultaneously.

        If successful results (see :py:attr:`Extract.success`) already exist in
        outdir, Pylada defaults to *not* repeating the calculations. In that
        case, the first object described above is *skipped* and only an
        extraction object is yielded.

        The :py:meth:`__call__` method loops over this generator and makes
        actual VASP_ calls. Looking at its code is a good place to start, if
        you want to understand this looping business. The benefit of this
        approach can be seen in :py:class:`~pylada.vasp.relax.iter_relax` (more
        complex calculations) and
        :py:class:`pylada.process.jobfolder.JobFolderProcess`. 

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
            Any attribute of the VASP instance can be overridden for
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
           
            This will never overwrite successfull VASP calculation, even if the
            parameters to the call are different.
    """ 
    ###from .. import vasp_program
    from ..process.program import ProgramProcess
    from .extract import Extract as ExtractVasp
    import os, sys, traceback

    if bugLev >= 5:
      #print 'vasp/functional iter: ===== start stack trace'
      #traceback.print_stack( file=sys.stdout)
      #print 'vasp/functional iter: ===== end stack trace'

      # Stack trace here is:
      #   ipython/launch/scattered_script.py", line 114, in <module>
      #     if __name__ == "__main__": main()
      #   ipython/launch/scattered_script.py", line 109, in main
      #     jobfolder[name].compute(comm=comm, outdir=name)
      #   jobfolder/jobfolder.py", line 295, in compute
      #     res = self.functional.__call__(**params)
      #
      #   Probably the following is the dynamically compiled code
      #   created by tools/makeclass: create_call_from_iter.
      #   Code is similar to:
      #     def __call__(self, structure, outdir=None, comm=None, **kwargs):
      #       result  = None
      #       for program in self.iter(structure, outdir=outdir,
      #           comm=comm, **kwargs):
      #         program.start(comm)
      #         program.wait()
      #       return result
      #   File "<string>", line 16, in __call__
      #
      #   Probably the following is the dynamically compiled code
      #   created by tools/makeclass: create_iter
      #   Code is similar to:
      #     def iter(self, structure, outdir=None, **kwargs):
      #       for o in iter_relax(
      #           SuperCall(self.__class__, self),
      #           structure, outdir=outdir, first_trial=self.first_trial,
      #           maxcalls=self.maxcalls, keepsteps=self.keepsteps,
      #           nofail=self.nofail, convergence=self.convergence,
      #           minrelsteps=self.minrelsteps, **kwargs):
      #         yield o
      #   File "<string>", line 12, in iter
      #
      #   vasp/relax.py", line 272, in iter_relax
      #    for u in vasp.iter( relaxed_structure, outdir = fulldir,
      #      restart = output, relaxation = relaxation, **params):
      #   vasp/functional.py", line 960, in iter
      #     traceback.print_stack( file=sys.stdout)

      print 'vasp/functional iter: outdir: %s' % (outdir,)
      print 'vasp/functional iter: structure:\n%s' % (structure,)

    # check for pre-existing and successful run.
    if not overwrite:
      # Check with this instance's Extract, cos it is this calculation we shall
      # do here. Derived instance's Extract might be checking for other stuff.
      extract = Vasp.Extract(outdir)
      if extract.success:
        yield extract # in which case, returns extraction object.
        return
    
    # copies/creates file environment for calculation.
    self.bringup(structure, outdir, comm=comm, overwrite=overwrite)

    # figures out what program to call.
    vaspProgram = getattr( self, 'program', None)
    if bugLev >= 5:
      print 'vasp/functional iter: vaspProgram: %s' % (vaspProgram,)
      print 'vasp/functional iter: testValidProgram: %s' % (testValidProgram,)
    if vaspProgram == None:
      raise RuntimeError('program was not set in the vasp functional')

    if testValidProgram != None: vaspProgram = testValidProgram

    if hasattr(vaspProgram, '__call__'):
      vaspProgram = vaspProgram(self, structure, comm)
    # creates a process with a callback to bring-down environment once it is
    # done.
    def onfinish(process, error):  self.bringdown(outdir, structure)
    onfail   = self.OnFail(Vasp.Extract(outdir))
    yield ProgramProcess( vaspProgram, cmdline=[], outdir=outdir,
			  onfinish=onfinish, onfail=onfail, stdout='stdout',
			  stderr='stderr', dompi=comm is not None )
    # yields final extraction object.
    yield ExtractVasp(outdir)

  def bringup(self, structure, outdir, **kwargs):
    """ Creates all input files necessary to run results.

        Performs the following actions.

        - Writes POSCAR_ file.
        - Writes INCAR_ file.
        - Writes KPOINTS_ file.
        - Creates POTCAR_ file
    """
    from os.path import join
    from ..crystal import specieset
    from ..misc.changedir import Changedir
    from . import files
    import os

    if bugLev >= 5:
      print 'vasp/functional bringup: outdir: ', outdir
      print 'vasp/functional bringup: structure:\n%s' % (structure,)
      print 'vasp/functional bringup: kwargs: ', kwargs

    with Changedir(outdir) as tmpdir:
      # creates INCAR file (and POSCAR via istruc).
      fpath = join(outdir, files.INCAR)
      if bugLev >= 5:
        print "vasp/functional bringup: incar fpath: ", fpath
      self.write_incar( structure, path=fpath, outdir=outdir, **kwargs )
  
      # creates kpoints file
      if bugLev >= 5:
        print "vasp/functional bringup: files.KPOINTS: ", files.KPOINTS
      with open(files.KPOINTS, "w") as kp_file: 
        self.write_kpoints(kp_file, structure)
  
      # creates POTCAR file
      if bugLev >= 5:
        print "vasp/functional bringup: files.POTCAR: ", files.POTCAR
      with open(files.POTCAR, 'w') as potcar:
        for s in specieset(structure):
          outLines = self.species[s].read_potcar()
          potcar.writelines( outLines)
      # Add is running file marker.
      with open('.pylada_is_running', 'w') as file: pass
      if bugLev >= 5:
        print 'vasp/functional bringup: is_run dir: %s' % (os.getcwd(),)
    
  def bringdown(self, directory, structure):
    """ Copies contcar to outcar. """
    from os.path import exists
    from os import remove
    from . import files
    from ..misc import Changedir
    import os, sys, traceback

    if bugLev >= 5:
      #print 'vasp/functional bringdown: ===== start stack trace'
      #traceback.print_stack( file=sys.stdout)
      #print 'vasp/functional bringdown: ===== end stack trace'

      # Stack trace here is:
      #   ipython/launch/scattered_script.py", line 114, in <module>
      #     if __name__ == "__main__": main()
      #   ipython/launch/scattered_script.py", line 109, in main
      #     jobfolder[name].compute(comm=comm, outdir=name)
      #   jobfolder/jobfolder.py", line 295, in compute
      #     res = self.functional.__call__(**params)
      #   File "<string>", line 25, in __call__
      #   process/program.py", line 352, in wait
      #     self.poll()
      #   process/program.py", line 201, in poll
      #     try: self.onfinish(process=self, error=(poll!=0))
      #   vasp/functional.py", line 994, in onfinish
      #     def onfinish(process, error):  self.bringdown(outdir, structure)
      #   vasp/functional.py", line 1059, in bringdown
      #     traceback.print_stack( file=sys.stdout)

      print 'vasp/functional bringdown: directory: ', directory
      print 'vasp/functional bringdown: initial structure:\n%s' \
        % (structure,)

    # Appends INCAR and CONTCAR to OUTCAR:

    # nomodoutcar
    #with Changedir(directory) as pwd:
    #  with open(files.OUTCAR, 'a') as outcar:
    #    if exists(files.CONTCAR):
    #      outcar.write('\n################ CONTCAR ################\n')
    #      with open(files.CONTCAR, 'r') as contcar: outcar.write(contcar.read())
    #      outcar.write('\n################ END CONTCAR ################\n')
    #    if exists(files.INCAR):
    #      outcar.write('\n################ INCAR ################\n')
    #      with open(files.INCAR, 'r') as incar: outcar.write(incar.read())
    #      outcar.write('\n################ END INCAR ################\n')
    #    outcar.write('\n################ INITIAL STRUCTURE ################\n')
    #    outcar.write("""from {0.__class__.__module__} import {0.__class__.__name__}\n"""\
    #                 """structure = {1}\n"""\
    #                 .format(structure, repr(structure).replace('\n', '\n            ')))
    #    outcar.write('\n################ END INITIAL STRUCTURE ################\n')
    #    if bugLev >= 5:
    #      print 'vasp/functional bringdown: initial structure written'
    #    outcar.write('\n################ FUNCTIONAL ################\n')
    #    outcar.write(repr(self))
    #    outcar.write('\n################ END FUNCTIONAL ################\n')

    with Changedir(directory) as pwd:
      # nomodoutcar
      with open('pylada.FUNCTIONAL', 'w') as fout:
        fout.write( repr( self))

      if exists('.pylada_is_running'): remove('.pylada_is_running')


  def write_incar(self, structure, path=None, **kwargs):
    """ Writes incar file. """
    from os.path import dirname
    from ..misc import RelativePath
    from .files import INCAR

    if bugLev >= 5:
      print "vasp/functional write_incar: path: ", path
      print "vasp/functional write_incar: structure:\n%s" % (structure,)
      print "vasp/functional write_incar: kwargs: ", kwargs

    # check what type path is.
    # if not a file, opens one an does recurrent call.
    if path is None: path = INCAR
    if not hasattr(path, "write"):
      with open(RelativePath(path).path, "w") as file:
        self.write_incar(structure, path=file, **kwargs)
      return
    if kwargs.get('outdir', None) is None:
      kwargs['outdir'] = dirname(path.name)

    self.output_map(structure=structure, vasp=self, **kwargs)
    # twice, in-case some parameters change others.

    # At this point (vasp==self)._input is a map of the incar parameters.
    # The values may be primitives or callables.
    # Example: self._input['foobar'] = 'FOOBARED'

    map = self.output_map(structure=structure, vasp=self, **kwargs)
    # The resulting map is a map of the incar parameters.
    # The values may be primitives or callables.

    length = max(len(u) for u in map)
    for key, value in map.iteritems():
      outLine = '{0: >{length}} = {1}\n'.format(
        key.upper(), value, length=length)
      path.write( outLine)



  def write_kpoints(self, file, structure, kpoints=None):
    """ Writes kpoints to a stream. """

    if bugLev >= 5:
      print "vasp/functional write_kpoints: file: ", file
      print "vasp/functional write_kpoints: structure:\n%s" % (structure,)
      print "vasp/functional write_kpoints: kpoints: ", kpoints

    if kpoints == None: kpoints = self.kpoints
    if isinstance(self.kpoints, str): file.write(self.kpoints)
    elif hasattr(self.kpoints, "__call__"):
      self.write_kpoints(file, structure, self.kpoints(self, structure))
    else: # numpy array or such.
      outLine = "Explicit list of kpoints.\n{0}\nCartesian\n".format(
        len(self.kpoints))
      file.write( outLine)

      for kpoint in self.kpoints:
        outLine = "{0[0]} {0[1]} {0[2]} {1}\n".format(
          kpoint, 1 if len(kpoint) == 3 else kpoint[3])
        file.write( outLine)

  def __repr__(self, defaults=True, name=None):
    """ Returns representation of this instance """
    from ..tools.uirepr import uirepr
    defaults = self.__class__() if defaults else None
    return uirepr(self, name=name, defaults=defaults)

  def __ui_repr__(self, imports, name=None, defaults=None, exclude=None):
    results = super(Vasp, self).__ui_repr__(imports, name, defaults, ['add_specie'])
    if name is None:
      name = getattr(self, '__ui_name__', self.__class__.__name__.lower())
    return results

  @add_setter
  def add_specie(self, args):
    """ Adds a specie to current functional. 
     
        The argument is a tuple containing the following.

        - Symbol/name (str). 
        - Directory where POTCAR_ resides (str).
        - List of U parameters (optional, see module vasp.specie).
        - Maximum (or minimum) oxidation state (optional, int).
        - ... Any other argument in order of `vasp.specie.Specie.__init__`.

        In practice, adding a specie to the dictionary :py:attr:`specie` can be
        done as follows:

        >>> vasp.add_specie = 'H', '$HOME/pseudos/PAWGGA/H'

        This would add an 'H' pseudo-potential to the mix. Subsequent calls to
        the functionals need *not* involve an H atom. It is possible to define
        more pseudo-potentials than are actually used. 
    """
    from .specie import Specie
    assert len(args) > 1, ValueError("Too few arguments.")
    self.species[args[0]] = Specie(*args[1:])
    
  def __setstate__(self, args):
    """ Sets state from pickle.

        Takes care of older pickle versions.
    """
    super(Vasp, self).__setstate__(args)
    for key, value in self.__class__().__dict__.iteritems():
       if not hasattr(self, key): setattr(self, key, value)

  class OnFail(object):
    """ Checks whether VASP run succeeded.

	Who knows what could happen and make VASP return an error. As long as
        the OUTCAR is OK, everything should be OK.
    """
    def __init__(self, extract):
      self.extract = extract
    def __call__(self, process, error):
      from ..process import Fail
      self.extract.uncache()
      try: success = self.extract.success
      except: success = False
      if not success:
        raise Fail( 'VASP failed to run correctly.\n'                       \
                    'It returned with error {0}.'.format(error) )

del stateless
del assign_attributes
del AttrBlock
del add_setter
del ExtractVasp
