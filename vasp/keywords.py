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

from quantities import eV
from ..tools.input import BoolKeyword as BaseBoolKeyword, ValueKeyword,        \
                          TypedKeyword, AliasKeyword, ChoiceKeyword,           \
                          BaseKeyword, QuantityKeyword
from pylada.misc import bugLev

class BoolKeyword(BaseBoolKeyword):
  """ Boolean keyword.

      If True, the keyword is present.
      If False, it is not.
  """
  def __init__(self, keyword=None, value=None):
    """ Initializes FullOptG keyword. """
    super(BoolKeyword, self).__init__(keyword=keyword, value=value)
  def output_map(self, **kwargs):
    """ Map keyword, value """
    if self.value is None: return None
    if getattr(self, 'keyword', None) is None: return None
    return { self.keyword: '.TRUE.' if self.value else '.FALSE.' }

class Magmom(ValueKeyword):
  """ Sets the initial magnetic moments on each atom.

      There are three types of usage: 

      - if None or False, does nothing
      - if calculations are not spin-polarized, does nothing.
      - if a string, uses that as for the MAGMOM_ keyword
      - if True and at least one atom in the structure has a non-zero
        ``magmom`` attribute, then creates the relevant moment input for VASP_

      If the calculation is **not** spin-polarized, then the magnetic moment
      tag is not set.

      .. note:: Please set by hand for non-collinear calculations

      .. seealso:: MAGMOM_

      .. _MAGMOM: http://cms.mpi.univie.ac.at/wiki/index.php/MAGMOM
  """
  keyword = 'MAGMOM'
  'VASP keyword'
  def __init__(self, value=None):
    super(Magmom, self).__init__(value=value)

  @property
  def value(self):
    """ MAGMOM value, or whether to compute it. 

        - if None or False, does nothing
        - if calculations are not spin-polarized, does nothing.
        - if a string, uses that as for the MAGMOM_ keyword
        - if True and at least one atom in the structure has a non-zero
          ``magmom`` attribute, then creates the relevant moment input for
          VASP_
    """
    return self._value
  @value.setter
  def value(self, value):
    if value is None: self._value = None
    elif value is True or value is False: self._value = value
    elif not isinstance(value, str):
      raise ValueError( 'Unknown value for magmom {0}. '                       \
                        'Should be True, False, None, or an adequate string'   \
                        .format(value) )
    else: self._value = value

  def output_map(self, **kwargs):
    """ MAGMOM input for VASP. """
    from ..crystal import specieset
    if self.value is None or self.value == False: return None
    if kwargs['vasp'].ispin == 1: return None
    if isinstance(self.value, str): return {self.keyword: str(self.value)}
    
    structure = kwargs['structure']
    if all(not hasattr(u, 'magmom') for u in structure): return None
    result = ""
    for specie in specieset(structure):

      # Example specie: Fe  moments: [-4.0, -4.0, -4.0, 4.0, -4.0, -4.0]
      moments = [getattr(u, 'magmom', 0e0) for u in structure if u.type == specie]
      # Get tupled = [[3, -4.0], [1, 4.0], [2, -4.0]]
      tupled = [[1, moments[0]]]
      for m in moments[1:]: 
        # Change precision from 1.e-12 to 1.e-1, per Vladan, 2013-10-09
        #if abs(m - tupled[-1][1]) < 1e-12: tupled[-1][0] += 1
        if abs(m - tupled[-1][1]) < 1e-1: tupled[-1][0] += 1
        else: tupled.append([1, m])

      # Create result = '3*-4.00 4.00 2*-4.00 '
      for i, m in tupled:
        # Change format from .2f to .1f, per Vladan, 2013-10-09
        if i == 1: result += "{0:.1f} ".format(m)
        else:      result += "{0}*{1:.1f} ".format(i, m)
    return {self.keyword: result.rstrip()}

class System(ValueKeyword):
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
  keyword = 'system'
  """ VASP keyword """
  def __init__(self, value=None):
    super(System, self).__init__(value=value)

  def output_map(self, **kwargs):
    """ Tries to return sensible title. 

        Never throws.
    """
    try: 
      if self.value is None:
        if len(getattr(kwargs['structure'], 'name', '')) == 0: return None
        return { self.keyword: str(kwargs['structure'].name) }
      elif len(getattr(kwargs['structure'], 'name', '')) == 0:
        try: return { self.keyword: str(self.value) }
        except: return None
      try:
        return { self.keyword:
                 '{0}: {1}'.format(str(self.value), kwargs['structure'].name) }
      except: return { self.keyword: str(kwargs['structure'].name) }
    except: return None

class Npar(ValueKeyword):
  """ Parallelization over bands. 

      Npar defines how many nodes work on one band.
      It can be set to a particular number:
  
      >>> vasp.npar = 2

      Or it can be deduced automatically. Different schemes are available:
      
        - power of two: npar is set to the largest power of 2 which divides the
          number of processors.
 
          >>> vasp.npar = "power of two"

          If the number of processors is not a power of two, prints nothing.

        - square root: npar is set to the square root of the number of processors.

          >>> vasp.npar = "sqrt"
      

      .. seealso: `NPAR <http://cms.mpi.univie.ac.at/vasp/guide/node138.html>`_
  """
  keyword = 'NPAR'
  """ VASP keyword """
  def __init__(self, value=None): super(Npar, self).__init__(value=value)

  def output_map(self, **kwargs):
    from math import log, sqrt
    if self.value is None: return None
    if not isinstance(self.value, str): 
      if self.value < 1: return None
      return {self.keyword: str(self.value)}

    comm = kwargs.get('comm', None)
    if comm is None: return None
    n = comm['n']
    if n == 1: return None
    if self.value == "power of two":
      m = int(log(n)/log(2))
      for i in range(m, 0, -1):
        if n % 2**i == 0: return {self.keyword: str(i)}
      return None
    if self.value == "sqrt":
      return {self.keyword: str(int(sqrt(n)+0.001))}
    raise ValueError("Unknown request npar = {0}".format(self.value))

class ExtraElectron(TypedKeyword):
  """ Sets number of electrons relative to neutral system.
      
      Gets the number of electrons in the (neutral) system. Then adds value to
      it and computes with the resulting number of electrons.

      >>> vasp.extraelectron =  0  # charge neutral system
      >>> vasp.extraelectron =  1  # charge -1 (1 extra electron)
      >>> vasp.extraelectron = -1  # charge +1 (1 extra hole)

      Disables :py:attr:`pylada.vasp.functional.Functional.nelect` if set to
      something other than None.

      .. seealso:: nelect_, NELECT_
      .. _NELECT: http://cms.mpi.univie.ac.at/wiki/index.php/NELECT
      .. _nelect: :py:attr:`~pylada.vasp.functional.Vasp.nelect`
  """
  type = float
  """ Type of this input. """
  keyword = 'nelect'
  """ VASP keyword. """
  def __init__(self, value=None): super(ExtraElectron, self).__init__(value=value)

  def __set__(self, instance, value):
    if value is not None: instance.nelect = None
    return super(ExtraElectron, self).__set__(instance, value)

  def nelectrons(self, vasp, structure):
    """ Total number of electrons in the system """
    from math import fsum
    # constructs dictionnary of valence charge
    valence = {}
    for key, value in vasp.species.items():
      valence[key] = value.valence
    # sums up charge.
    return fsum( valence[atom.type] for atom in structure )
    
  def output_map(self, **kwargs):
    if self.value is None: return None
    if self.value == 0: return None
    # gets number of electrons.
    charge_neutral = self.nelectrons(kwargs['vasp'], kwargs['structure'])
    # then prints incar string.
    return {self.keyword: str(charge_neutral + self.value)}

class NElect(TypedKeyword):
  """ Sets the absolute number of electrons.
      
      Disables :py:attr:`pylada.vasp.functional.Functional.extraelectron` if set to
      something other than None.

      .. seealso:: extraelectron_, NELECT_
      .. _NELECT: http://cms.mpi.univie.ac.at/wiki/index.php/NELECT
      .. _extraelectron: :py:attr:`~pylada.vasp.functional.Vasp.extraelectron`
  """
  type = float
  """ Type of this input. """
  keyword = 'nelect'
  """ VASP keyword. """
  def __init__(self, value=None): super(NElect, self).__init__(value=value)

  def __set__(self, instance, value):
    if value is not None: instance.extraelectron = None
    return super(NElect, self).__set__(instance, value)

class Algo(ValueKeyword): 
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

      If :py:data:`is_vasp_4 <pylada.is_vasp_4>` is an existing configuration
      variable of :py:mod:`pylada` the parameters marked as vasp 5 will fail.

      .. warning:: The string None is not  allowed, as it would lead to
         confusion with the python object None. Please use "Nothing" instead.
         The python object None will simply not print the ALGO keyword to the
         INCAR file.

      .. note:: By special request, "fast" is the default algorithm.

      .. seealso:: `ALGO <http://cms.mpi.univie.ac.at/vasp/vasp/ALGO_tag.html>`_
  """ 
  keyword = 'algo'
  """ VASP keyword. """
  def __init__(self, value="fast"): super(Algo, self).__init__(value=value)
  @property
  def value(self): return self._value
  @value.setter
  def value(self, value):
    if value is None: self._value = None; return None
    try: from pylada import is_vasp_4
    except: is_vasp_4 = False
    if not hasattr(value, 'lower'):
      raise TypeError("ALGO cannot be set with {0}.".format(value))
    lower = value.lower().rstrip().lstrip()
    lower = lower.replace('_', '')
    lower = lower.replace('-', '')
    if is_vasp_4                                                               \
       and ( lower[0] in ['c', 's', 'e']                                       \
             or lower in [ "nothing", "subrot", "exact",                       \
                           "gw", "gw0", "chi", "scgw",                         \
                           "scgw0"] ): 
      raise ValueError("algo value ({0}) is not valid with VASP 4.6.".format(value))

    if lower == "diag": value = "Diag"
    elif lower == "nothing": value = "Nothing"
    elif lower == "chi":  value = "chi"
    elif lower == "gw":   value = "GW"
    elif lower == "gw0":  value = "GW0"
    elif lower == "scgw": value = "scGW"
    elif lower == "scgw0": value = "scGW0"
    elif lower[0] == 'v': value = "Very_Fast" if is_vasp_4 else 'VeryFast'
    elif lower[0] == 'f': value = "Fast"
    elif lower[0] == 'n': value = "Normal"
    elif lower[0] == 'd': value = "Damped"
    elif lower[0] == 'a': value = "All"
    elif lower[0] == 'c': value = "Conjugate"
    elif lower[0] == 's': value = "Subrot"
    elif lower[0] == 'e': value = "Eigenval"
    else:
      self._value = None
      raise ValueError("algo value ({0!r}) is invalid.".format(value))
    self._value = value

class Ediff(TypedKeyword):
  """ Sets the absolute energy convergence criteria for electronic minimization.

      EDIFF_ is set to this value in the INCAR. 

      Sets ediff_per_atom_ to None.

      .. seealso:: EDIFF_, ediff_per_atom_
      .. _EDIFF: http://cms.mpi.univie.ac.at/wiki/index.php/EDIFFG
      .. _ediff_per_atom: :py:attr:`~pylada.vasp.functional.Vasp.ediff_per_atom`
  """
  type = float
  """ Type of the value """
  keyword = 'ediff'
  """ VASP keyword """
  def __init__(self, value=None):
    """ Creates *per atom* tolerance. """
    super(Ediff, self).__init__(value=value)
  def __set__(self, instance, value):
    if value is None: 
      self.value = None 
      return
    instance.ediff_per_atom = None
    if value < 0e0: value = 0
    return super(Ediff, self).__set__(instance, value)

class EdiffPerAtom(TypedKeyword):
  """ Sets the relative energy convergence criteria for electronic minimization.

      EDIFF_ is set to this value *times* the number of atoms in the structure.
      This approach is more sensible than straight-off ediff_ when doing
      high-throughput over many structures.

      Sets ediff_ to None.

      .. seealso:: EDIFF_, ediff_

      .. _EDIFF: http://cms.mpi.univie.ac.at/wiki/index.php/EDIFFG

      .. _ediff: :py:attr:`~pylada.vasp.functional.Vasp.ediff`
  """
  type = float
  """ Type of the value """
  keyword = 'ediff'
  """ VASP keyword """
  def __init__(self, value=None):
    """ Creates *per atom* tolerance. """
    super(EdiffPerAtom, self).__init__(value=value)
  def __set__(self, instance, value):
    if value is None: 
      self.value = None 
      return
    instance.ediff = None
    if value < 0e0: value = 0
    return super(EdiffPerAtom, self).__set__(instance, value)
  def output_map(self, **kwargs):
    if self.value is None: return 
    return { self.keyword: str(self.value * float(len(kwargs["structure"]))) }

class Ediffg(TypedKeyword):
  """ Sets the absolute energy convergence criteria for ionic relaxation.

      EDIFFG_ is set to this value in the INCAR. 

      Sets ediffg_per_atom_ to None.

      .. seealso:: EDIFFG_, ediffg_per_atom_
      .. _EDIFFG: http://cms.mpi.univie.ac.at/vasp/guide/node105.html
      .. _ediffg_per_atom: :py:attr:`~pylada.vasp.functional.Vasp.ediffg_per_atom`
  """
  type = float
  """ Type of the value """
  keyword = 'ediffg'
  """ VASP keyword """
  def __init__(self, value=None):
    """ Creates *per atom* tolerance. """
    super(Ediffg, self).__init__(value=value)
  def __set__(self, instance, value):
    if value is None: 
      self.value = None 
      return
    instance.ediffg_per_atom = None
    return super(Ediffg, self).__set__(instance, value)

class EdiffgPerAtom(TypedKeyword):
  """ Sets the relative energy convergence criteria for ionic relaxation.

      - if positive: EDIFFG_ is set to this value *times* the number of atoms
        in the structure. This means that the criteria is for the total energy per atom.
      - if negative: same as a negative EDIFFG_, since that convergence
        criteria is already per atom.
      
      This approach is more sensible than straight-off ediffg_ when doing
      high-throughput over many structures.

      Sets ediffg_ to None.

      .. seealso:: EDIFFG_, ediff_
      .. _EDIFFG: http://cms.mpi.univie.ac.at/wiki/index.php/EDIFFG
      .. _ediffg: :py:attr:`~pylada.vasp.functional.Vasp.ediffg`
  """
  type = float
  """ Type of the value """
  keyword = 'ediffg'
  """ VASP keyword """
  def __init__(self, value=None):
    """ Creates *per atom* tolerance. """
    super(EdiffgPerAtom, self).__init__(value=value)
  def __set__(self, instance, value):
    if value is None: 
      self.value = None 
      return
    instance.ediffg = None
    return super(EdiffgPerAtom, self).__set__(instance, value)
  def output_map(self, **kwargs):
    if self.value is None: return 
    if self.value > 0e0:
      return { self.keyword: str(self.value * float(len(kwargs["structure"]))) }
    else: return { self.keyword: str(self.value) }

class Encut(ValueKeyword):
  """ Defines cutoff factor for calculation. 

      There are three ways to set this parameter:

      - if value is floating point and 0 < value <= 3: then the cutoff is
        ``value * ENMAX``, where ENMAX is the maximum recommended cutoff for
        the species in the system.
      - if value > 3 eV, then prints encut is exactly value (in eV). Any energy
        unit is acceptable.
      - if value < 0 eV or None, does not print anything to INCAR. 
      
      .. seealso:: `ENCUT <http://cms.mpi.univie.ac.at/vasp/vasp/ENCUT_tag.html>`_
  """
  keyword = "encut"
  """ Corresponding VASP key. """
  def __init__(self, value=None): super(Encut, self).__init__(value=value)

  def output_map(self, **kwargs):
    from quantities import eV
    from ..crystal import specieset
    value = self.value
    if hasattr(self.value, 'units'):
      value = self.value.rescale(eV).magnitude
      return {self.keyword: str(value)} if value > 1e-12 else None
    if value is None:   return None
    elif value < 1e-12: return None
    elif value >= 1e-12 and value <= 3.0:
      types = specieset(kwargs["structure"])
      encut = max(kwargs["vasp"].species[type].enmax for type in types)
      if hasattr(encut, 'rescale'): encut = float(encut.rescale(eV))
      return {self.keyword: str(encut * value)}
    return {self.keyword: str(value)}

##? gwmod?
class EncutGW(Encut):
  """ Defines cutoff factor for GW calculation. 

      There are three ways to set this parameter:

      - if value is floating point and 0 < value <= 3: then the cutoff is
        ``value * ENMAX``, where ENMAX is the maximum recommended cutoff for
        the species in the system.
      - if value > 3 eV, then prints encut is exactly value (in eV). Any energy
        unit is acceptable.
      - if value < 0 eV or None, does not print anything to INCAR. 
      
      .. seealso:: `ENCUTGW
        <http://cms.mpi.univie.ac.at/vasp/vasp/ENCUTGW_energy_cutoff_response_function.html>`_
  """
  keyword = 'encutgw'

class ICharg(AliasKeyword):
  """ Charge from which to start. 

      It is best to keep this attribute set to -1, in which case, Pylada takes
      care of copying the relevant files.

        - -1: (Default) Automatically determined by Pylada. Depends on the value
              of restart_ and the existence of the relevant files. Also takes
              care of non-scf bit.
  
        - 0: Tries to restart from wavefunctions. Uses the latest WAVECAR file
             between the one currently in the output directory and the one in
             the restart directory (if specified). Sets nonscf_ to False.
  
             .. note:: CHGCAR is also copied, just in case.
  
        - 1: Tries to restart from wavefunctions. Uses the latest WAVECAR file
             between the one currently in the output directory and the one in
             the restart directory (if specified). Sets nonscf_ to False.
  
        - 2: Superimposition of atomic charge densities. Sets nonscf_ to False.
  
        - 4: Reads potential from POT file (VASP-5.1 only). The POT file is
             deduced the same way as for CHGAR and WAVECAR above.  Sets nonscf_
             to False.
  
        - 10, 11, 12: Same as 0, 1, 2 above, but also sets nonscf_ to True. This
             is a shortcut. The value is actually kept to 0, 1, or 2:
  
             >>> vasp.icharg = 10
             >>> vasp.nonscf, vasp.icharg
             (True, 0)

      .. note::
      
         Files are copied right before the calculation takes place, not before.

      .. seealso:: ICHARG_, nonscf_, restart_, istruc_, istart_

      .. _ICHARG: http://cms.mpi.univie.ac.at/wiki/index.php/ICHARG

      .. _nonscf: :py:attr:`~pylada.vasp.functional.Functional.nonscf`
      .. _restart: :py:attr:`~pylada.vasp.functional.Functional.restart`
      .. _istruc: :py:attr:`~pylada.vasp.functional.Functional.istruc`
      .. _istart: :py:attr:`~pylada.vasp.functional.Functional.istart`
  """ 
  keyword = 'icharg'
  """ VASP keyword """
  aliases = { -1: ['auto', -1],
               0: ['wfns', 'wavefunction', 'wavefunctions', 'wfn', 0],
               1: ['chgcar', 'CHGCAR', 'file', 1],
               2: ['atomic', 'scratch', 2],
               4: ['pot', 4],
               10: [10], 11: [11], 12: [12], 14: [14] }
  """ Mapping of aliases. """
  def __init__(self, value='auto'): 
    super(ICharg, self).__init__(value=value)
  def __set__(self, instance, value):
    """ Sets internal value. 

        Makes sure that the input value is allowed, and that the nonscf_
        attribute is set properly.

      .. _nonscf: :py:attr:`~pylada.vasp.functional.Functional.nonscf`
    """
    super(ICharg, self).__set__(instance, value)
    if self._value is None: return
    if self._value > 10: 
      self._value -= 10
      instance.nonscf = True
    elif self._value >= 0:
      instance.nonscf = False

  def output_map(self, **kwargs):
    from os.path import join
    from ..misc import latest_file, copyfile
    from ..error import ValueError
    from . import files

    icharge = self._value
    if icharge is None: return None

    # some files will be copied.
    if icharge not in [2, 12]: 
      # determines directories to look into.
      vasp = kwargs['vasp']
      outdir = kwargs['outdir']
      hasrestart = getattr(vasp.restart, 'success', False)
      directories = [outdir]
      if hasrestart: directories += [vasp.restart.directory]
      # determines which files exist
      last_wfn = None if icharge in [1, 11]                                    \
                 else latest_file( *[ join(u, files.WAVECAR) 
                                      for u in directories ] )
      last_chg = latest_file(*[join(u, files.CHGCAR) for u in directories]) 
      last_pot = None if icharge not in [-1, 4, 14]                            \
                 else latest_file(*[join(u, files.POT) for u in directories])

      # determines icharge depending on file. 
      if icharge < 0:
        if last_wfn is not None: icharge = 10 if vasp.nonscf else 0
        elif last_chg is not None: icharge = 11 if vasp.nonscf else 1
        elif last_pot is not None: icharge = 4 if vasp.nonscf else 14
        else: icharge = 2
      elif icharge == 1 and last_chg is None:
        raise ValueError('CHGCAR could not be found, yet ISTART=1 requested.')

      # copies relevant files.
      if icharge in [0, 10] and last_wfn is not None:
        copyfile(last_wfn, outdir, nothrow='same')
      if icharge in [0, 1, 10, 11, 4, 14] and last_chg is not None:
        copyfile(last_chg, outdir, nothrow='same')
      if icharge in [4, 14] and last_pot is not None:
        copyfile(last_pot, outdir, nothrow='same')
    return {self.keyword: str(icharge)}

class IStart(AliasKeyword):
  """ Starting wavefunctions.

      It is best to keep this attribute set to -1, in which case, Pylada takes
      care of copying the relevant files.

        - -1: Automatically determined by Pylada. Depends on the value of restart_
              and the existence of the relevant files.
  
        - 0: Start from scratch.

        - 1: Restart with constant cutoff.
  
        - 2: Restart with constant basis.
  
        - 3: Full restart, including TMPCAR.

      .. note::
      
         Files are copied right before the calculation takes place, not before.

      .. seealso:: ISTART_, icharg_, istruc_, restart_
      .. _ISTART: http://cms.mpi.univie.ac.at/wiki/index.php/ISTART
      .. _restart: :py:attr:`~pylada.vasp.functional.Functional.restart`
      .. _icharg: :py:attr:`~pylada.vasp.functional.Functional.icharg`
      .. _istruc: :py:attr:`~pylada.vasp.functional.Functional.istruc`
  """ 
  keyword = 'istart'
  """ VASP keyword """
  aliases = { -1: ['auto', -1], 0: ['scratch', 0],
               1: ['cutoff', 1], 2: ['basis', 2], 3: ['tmpcar', 'full', 3] }
  """ Mapping of aliases. """
  def __init__(self, value=-1): 
    super(IStart, self).__init__(value=value)

  def output_map(self, **kwargs):
    from os.path import dirname, join
    from ..misc import latest_file, copyfile
    from ..error import ValueError
    from . import files

    istart = self._value
    if istart is None: return None
    # some files will be copied.
    if istart != 0:
      # determines directories to look into.
      vasp = kwargs['vasp']
      outdir = kwargs['outdir']
      hasrestart = getattr(vasp.restart, 'success', False)
      directories = [outdir]
      if hasrestart: directories += [vasp.restart.directory]
      # determines which files exist
      last_wfn = latest_file(*[join(u, files.WAVECAR) for u in directories])
      # Validity of TMPCAR depends on existence of WAVECAR
      if last_wfn is None: last_tmp = None
      else: last_tmp = latest_file(join(dirname(last_wfn), files.TMPCAR))

      # determines icharge depending on file. 
      if last_wfn is None and istart > 0: 
        raise ValueError( 'Wavefunction does not exist, '                      \
                          'yet ISTART={0} requested.'.format(istart) )
      if last_tmp is None and istart == 3:
        raise ValueError( 'TMPCAR file does not exist and ISTART={0}'          \
                          .format(istart) )
      if last_wfn is not None: copyfile(last_wfn, outdir, nothrow='same')
      if last_tmp is not None: copyfile(last_tmp, outdir, nothrow='same')
      if istart == -1:
        if last_wfn is not None and last_tmp is not None: istart = 3
        elif last_wfn is not None: istart = 1
        else: istart = 0
    return {self.keyword: str(istart)}

class IStruc(AliasKeyword):
  """ Initial structure. 
  
      Determines which structure is written to the POSCAR. In practice, it
      makes it possible to restart a crashed job from the latest CONTCAR_.
      There are two possible options:

        - auto: 
        
          Pylada determines automatically what to use. The structure can be read
          from the following files, in order of priority:

            - CONTCAR of the current
            - OUTCAR of the restart 

          If ``overwrite==True`` when calling the vasp functional, then never
          read from CONTCAR.

        - scratch: Always uses input/restart structure.
        - contcar: Always use CONTCAR_ structure unless ``overwrite == True``.
        - input: Always use input structure, never restart_ or CONTCAR_ structure.

      Only positions and cells are used from CONTCAR_. All other attributes
      (magnetization, etc) are those of the input structure. Hence, the atoms
      of any given specie should be same order in the CONTCAR and in the
      input/restart.

      .. note:: There is no VASP equivalent to this option.
      .. seealso:: :py:attr:`restart`, :py:attr:`icharg`, :py:attr:`istart`

      .. _CONTCAR: http://cms.mpi.univie.ac.at/vasp/guide/node60.html
  """
  aliases = {-1: ['auto', 0], 0: ['scratch', 1], 1: ['contcar', 1], 2: ['input', 2] }
  """ Aliases for the same option. """
  keyword = None
  """ Does not correspond to a VASP keyword """

  def __init__(self, value='auto'):
    if bugLev >= 5:
      print 'keywords: IStruct.init for CONTCAR: value: %s' % (value,)
    super(IStruc, self).__init__(value=value)

  def output_map(self, **kwargs):
    from os.path import join
    from ..misc import latest_file
    from ..error import ValueError
    from ..crystal import write, read, specieset
    from . import files

    if bugLev >= 5:
      print 'keywords: IStruct.output for CONTCAR: _value: %s' \
        % (self._value,)
    istruc = self._value
    if istruc is None: istruc = 0
    if kwargs.get('overwrite', False): istruc = 0
    structure = kwargs['structure']
    outdir = kwargs['outdir']
    vasp = kwargs['vasp']
    has_restart = getattr(vasp, 'restart', None) is not None and istruc != 2
    if has_restart: has_restart = vasp.restart.success
    if has_restart: structure = vasp.restart.structure
    if bugLev >= 5:
      print 'keywords: IStruct.output: istruc: %s' % (istruc,)
      print 'keywords: IStruct.output: has_restart: %s' % (has_restart,)
      print 'keywords: IStruct.output: outdir: %s' % (outdir,)
      print 'keywords: IStruct.output: structure:\n%s' % (structure,)

    # determines which CONTCAR is the latest, if any exist.
    if istruc in [-1, 1]: 
      last_contcar = latest_file(join(outdir, files.CONTCAR))
      if bugLev >= 5:
        print 'keywords: IStruct.output: last_contcar: %s' % (last_contcar,)
      # if a contcar exists and we should re-read, then modifies structure
      # accordingly. It is expected that the structures are equivalent, in the
      # sense that they have the same atoms in the same order (more
      # specifically, the order of the atoms of a given specie are the same). 
      if last_contcar is not None:
        other = read.poscar(path=last_contcar, types=specieset(structure))
        if len(other) != len(structure):
          raise ValueError('CONTCAR and input structure differ in size.')
        for type in specieset(other):
          A = [a for a in structure if a.type == type]
          B = [a for a in other if a.type == type]
          if len(A) != len(B): 
            raise ValueError( 'CONTCAR and input structure differ '            \
                              'in number of {0} atoms'.format(type) )
          for a, b in zip(A, B):
            if a.type != type: continue
            a.pos = b.pos
        structure.cell = other.cell
        structure.scale = other.scale
        if bugLev >= 5:
          print 'keywords: IStruct.output: outdir: %s' % (outdir,)
          print 'keywords: IStruct.output: new structure:\n%s' % (structure,)

    # Depending on different options and what's available, writes structure or
    # copies contcar.
    if len(structure) == 0: raise ValueError('Structure is empty')
    if structure.scale < 1e-8: raise ValueError('Structure scale is zero')
    if structure.volume < 1e-8: raise ValueError('Structure volume is zero')
    write.poscar(structure, join(outdir, 'POSCAR'))
    return None

class LDAU(BoolKeyword): 
  """ Sets U, nlep, and enlep parameters. 
 
      The U, nlep, and enlep parameters of the atomic species are set at the
      same time as the pseudo-potentials. This object merely sets up the incar
      with the right input if the species are defined with U or NLEP_ parameters. 

      *If* there are species with NLEP parameters, then:

      >>> vasp.ldau = True

      will add the right input to the incar (``vasp.ldau`` is True by default).
      If there are no such species, then the line above will *not* result in
      adding anything to the incar. 

      .. note:: NLEP_ requires VASP to be patched for it. Furthermore, it
         requires vasp_has_nlep_ to set to True (False by default) in your
         pylada configuration file.

      .. seealso:: LDAU_, LDAUTYPE_, LDAUL_, LDAUJ_

      .. _LDAU: http://cms.mpi.univie.ac.at/wiki/index.php/LDAU
      .. _LDAUTYPE: http://cms.mpi.univie.ac.at/wiki/index.php/LDAUTYPE
      .. _LDAUL: http://cms.mpi.univie.ac.at/wiki/index.php/LDAUL
      .. _LDAUU: http://cms.mpi.univie.ac.at/wiki/index.php/LDAUU
      .. _LDAUJ: http://cms.mpi.univie.ac.at/wiki/index.php/LDAUJ
      .. _NLEP: http://prb.aps.org/abstract/PRB/v77/i24/e241201
  """
  keyword = 'LDAU'
  """ VASP keyword corresponding to the value. """
  def __init__(self, value=True): super(LDAU, self).__init__(value=value)

  def output_map(self, **kwargs):
    from ..crystal import specieset
    from ..error import ValueError, ConfigError, internal
    ###from .. import vasp_has_nlep
    vasp = kwargs['vasp']
    has_nlep = getattr( vasp, 'has_nlep', False)
    if bugLev >= 5:
      print 'vasp/keywords: ldau.output_map:'
      print '    has_nlep: %s' % (has_nlep,)

    #print "vasp/keywords: LDAU.output: keyword: %s  value: %s" \
    #  % (self.keyword, self.value,)
    if self.value is None or self.value is False: return
    types = specieset(kwargs['structure'])
    species = kwargs['vasp'].species
    # existence and sanity check
    has_U, which_type = False, None 
    for type in types:
      specie = species[type]
      if bugLev >= 5:
        print '        type: %s  specie: %s  U: %s  len: %d' \
          % (type, specie, specie.U, len(specie.U),)
        # Shows, for Cu2 Al2 O4:
        #   type: Cu
        #   specie: Specie('/nopt/nrel/ecom/cid/vasp.pseudopotentials.a/
        #     pseudos/Cu', oxidation=1, U=[{'type': 2, 'J': 0.0, 'U': 5.0,
        #     'l': 2, 'func': 'U'}], moment=[1.0, 4.0])
        #   U: [{'type': 2, 'J': 0.0, 'U': 5.0, 'l': 2, 'func': 'U'}]
        #   len: 1
        #   ... similarly for Al, O
      if len(specie.U) == 0: continue
      if len(specie.U) > 4: 
        raise ValueError("More than 4 channels for U/NLEP parameters")
      has_U = True
      # check whether running NLEP without NLEP VASP.
      if not has_nlep:
        if len(specie.U) > 1:
          raise ValueError('has_nlep is False. There can be only U parameter')
        if specie.U[0]['func'] != 'U': 
          raise ConfigError('has_nlep is False. Cannot use NLEP.')
      # checks consistency.
      which_type = specie.U[0]["type"]
      if bugLev >= 5:
        print '        which_type: %s' % (which_type,)
        # Shows: which_type: 2
      for l in specie.U[1:]: 
        if which_type != l["type"]:
          raise ValueError("LDA+U/NLEP types are not consistent across species.")
    if not has_U: return None

    # parameters other than U and NLEP themselves.
    result = super(LDAU, self).output_map(**kwargs)
    result['LDAU'] = '.TRUE.'
    result['LDAUTYPE'] = str(which_type)
    if bugLev >= 5:
      print '        initial result: %s' % (result,)
      # Shows: initial result: {'LDAU': '.TRUE.', 'LDAUTYPE': '2'}

    # U and NLEP themselves.
    if has_nlep: 
      if bugLev >= 5: print '  has_nlep is True'
      for i in range( max(len(species[type].U) for type in types) ):
        ldul, lduu, lduj, lduo = [], [], [], []
        for type in types:
          specie = species[type]
          # a = 4 tuple, for func=='U':
          #   ldul component = U['l'], default = -1
          #   lduu component = U['U'], default = 0
          #   lduj component = U['J'], default = 0
          #   lduo component = 1
          a = -1, 0e0, 0e0, 1
          if len(specie.U) <= i: pass
          elif specie.U[i]["func"] == "U":    
            a = [specie.U[i]["l"], specie.U[i]["U"], specie.U[i]["J"], 1]
          elif specie.U[i]["func"] == "nlep": 
            a = [specie.U[i]["l"], specie.U[i]["U0"], 0e0, 2]
          elif specie.U[i]["func"] == "enlep":
            a = [specie.U[i]["l"], specie.U[i]["U0"], specie.U[i]["U1"], 3]
          else: raise internal("Debug Error.")
          if hasattr(a[1], "rescale"): a[1] = a[1].rescale("eV")
          if hasattr(a[2], "rescale"): a[2] = a[2].rescale("eV")
          ldul.append('{0[0]}'.format(a))
          lduu.append('{0[1]:18.10e}'.format(a))
          lduj.append('{0[2]:18.10e}'.format(a))
          lduo.append('{0[3]}'.format(a))
          if bugLev >= 5:
            print '      i: %d  type: %s  specie: %s  U: %s' \
              % (i, type, specie, specie.U,)
            print '          a: %s' % (a,)
            print '          ldul: %s' % (ldul,)
            print '          lduu: %s' % (lduu,)
            print '          lduj: %s' % (lduj,)
            print '          lduo: %s' % (lduo,)
            # Shows, for Cu2 Al2 O4 (reformatted):
            #   i: 0  type: Cu
            #   specie: Specie('.../pseudos/Cu', oxidation=1, U=[{'type': 2, 'J': 0.0, 'U': 5.0, 'l': 2, 'func': 'U'}], moment=[1.0, 4.0])
            #   U: [{'type': 2, 'J': 0.0, 'U': 5.0, 'l': 2, 'func': 'U'}]
            #   a: [2, 5.0, 0.0, 1]
            #   ldul: ['2']
            #   lduu: ['  5.0']
            #   lduj: ['  0.0']
            #   lduo: ['1']
            #
            #   i: 0  type: Al
            #   specie: Specie('.../pseudos/Al', oxidation=3)
            #   U: []
            #   a: (-1, 0.0, 0.0, 1)
            #   ldul: ['2', '-1']
            #   lduu: ['  5.0', '  0.0']
            #   lduj: ['  0.0', '  0.0']
            #   lduo: ['1', '1']
            #
            #   i: 0  type: O
            #   specie: Specie('.../pseudos/O', oxidation=-2)
            #   U: []
            #   a: (-1, 0.0, 0.0, 1)
            # Parallel arrays:
            #          Cu      Al      O
            #   ldul: ['2',    '-1',   '-1']    # from specie.U['l']
            #   lduu: ['5.0',  '0.0',  '0.0']   # from specie.U['U']
            #   lduj: ['0.0',  '0.0',  '0.0']   # from specie.U['j']
            #   lduo: ['1',    '1',    '1']     # ==func=='U'

        result['LDUL{0}'.format(i+1)] = ' '.join(ldul)
        result['LDUU{0}'.format(i+1)] = ' '.join(lduu)
        result['LDUJ{0}'.format(i+1)] = ' '.join(lduj)
        result['LDUO{0}'.format(i+1)] = ' '.join(lduo)
        if bugLev >= 5:
          print '  has_nlep result: %s' % (result,)
          # Shows:
          #   has_nlep result: {
          #     'LDUO1': '1 1 1',
          #     'LDUU1': '  5.0  0.0   0.0',
          #     'LDUJ1': '  0.0  0.0   0.0',
          #     'LDUL1': '2 -1 -1',
          #     'LDAUTYPE': '2',
          #     'LDAU': '.TRUE.'}
    else: 
      if bugLev >= 5: print '  Not has_nlep'
      ldul, lduu, lduj = [], [], []
      for type in types:
        specie = species[type]
        a = -1, 0e0, 0e0, 1
        if len(specie.U) <= 0: pass
        elif specie.U[0]["func"] == "U":    
          a = [specie.U[0]["l"], specie.U[0]["U"], specie.U[0]["J"], 1]
        if hasattr(a[1], "rescale"): a[1] = a[1].rescale("eV")
        if hasattr(a[2], "rescale"): a[2] = a[2].rescale("eV")
        ldul.append('{0[0]}'.format(a))
        lduu.append('{0[1]:18.10e}'.format(a))
        lduj.append('{0[2]:18.10e}'.format(a))
        if bugLev >= 5:
          print '        type: %s  specie: %s  U: %s' \
            % (type, specie, specie.U,)
          print '          ldul: %s' % (ldul,)
          print '          lduu: %s' % (lduu,)
          print '          lduj: %s' % (lduj,)
          print '          lduo: %s' % (lduo,)
      result['LDUL'] = ' '.join(ldul)
      result['LDUU'] = ' '.join(lduu)
      result['LDUJ'] = ' '.join(lduj)
      if bugLev >= 5:
        print '  Not has_nlep final result: %s' % (result,)
    return result

class PrecFock(AliasKeyword):
  aliases = { 'Low': ['low'], 'Medium': ['medium'], 'Fast': ['fast'],
              'Normal': ['normal'], 'Accurate': ['accurate'] }
  """ Aliases for the values of the VASP keyword. """
  keyword = 'PRECFOCK'
  """ Vasp keyword. """

class Precision(AliasKeyword):
  aliases = { 'Accurate': ['Accurate', 'accurate'],
              'Low': ['Low', 'low'],
              'Normal': ['Normal', 'normal'],
              'Medium': ['Medium', 'medium'],
              'High': ['High', 'high'],
              'Single': ['Single', 'single'] }
  """ Aliases for the values of the VASP keyword. """
  keyword = 'PREC'
  """ Vasp keyword. """


class Nsw(TypedKeyword):
  type = int
  """ Type of the keyword. """
  keyword = 'nsw'
  """ VASP keyword. """


class Isif(ChoiceKeyword):
  keyword = 'isif'
  values = range(8)


class IBrion(BaseKeyword):
  keyword = 'ibrion'
  """ VASP keyword """
  def __init__(self, value=None):
    super(IBrion, self).__init__()
    self.value = value
  def __get__(self, instance, owner=None): return self.value
  def __set__(self, instance, value):
    from ..error import ValueError
    try: dummy = int(value)
    except: raise ValueError('ibrion accepts only integer values')
    else: value = dummy
    if value < -1 or (value > 8 and value !=44):
      raise ValueError('Unexpected value for IBRION')
    self.value = value
  def output_map(self, **kwargs):
    vasp = kwargs['vasp']
    if vasp.relaxation == 'static': 
      return {self.keyword: str(-1)}
    if self.value is None: return None
    return {self.keyword: str(self.value)}


class Relaxation(BaseKeyword):
  """ Simple relaxation parameter 

      It accepts two parameters:

        - static: for calculation without geometric relaxation.
        - combination of ionic, volume, cellshape: for the type of relaxation
          requested.

      It makes sure that isif_, ibrion_, and nsw_ take the right value for the
      kind of relaxation.
  """
  keyword = None
  """ Just an alias for ISIF. """


  def __init__(self, value=None):
    super(Relaxation, self).__init__()
    self.value = value
    if bugLev >= 5:
      print 'keywords: Relaxation.init: value: %s' % (self.value,)


  def __get__(self, instance, owner=None): 
    if bugLev >= 5:
      print 'keywords: Relaxation.get: nsw: %s' % (instance.nsw,)
      print 'keywords: Relaxation.get: ibrion: %s' % (instance.ibrion,)
      print 'keywords: Relaxation.get: isif: %s' % (instance.isif,)
    nsw = instance.nsw if instance.nsw is not None else 0
    ibrion = instance.ibrion if instance.ibrion is not None                    \
             else (-1 if nsw <= 0 else 2)
    if nsw <= 0 or ibrion == -1: return 'static'
    return { None: 'ionic',
             0: 'ionic', 
             1: 'ionic', 
             2: 'ionic', 
             3: 'cellshape ionic volume',
             4: 'cellshape ionic',
             5: 'cellshape',
             6: 'cellshape volume',
             7: 'volume' }[instance.isif]


  def __set__(self, instance, value):
    import sys, traceback
    from ..error import ValueError
    if bugLev >= 5:
      print 'keywords: Relaxation.set A: value: %s' % (value,)
      # Shows: volume ionic cellshape
      #print 'keywords: Relaxation.set: ===== start stack trace'
      #traceback.print_stack( file=sys.stdout)
      #print 'keywords: Relaxation.set: ===== end stack trace'

      # Stack trace A, from ipython script, as part of tst.nonmagnetic_wave()
      # File "test.py", line 58, in nonmagnetic_wave
      #   input = read_input(inputpath)
      # File "vasp/__init__.py", line 67, in read_input
      #   return read_input(filepath, input_dict)
      # File "misc/__init__.py", line 195, in read_input
      #   return exec_input(string, global_dict, local_dict,
      #     paths, basename(filename))
      # File "misc/__init__.py", line 226, in exec_input
      #   exec(script, global_dict, local_dict)
      # File "<string>", line 160, in <module>
      # File "tools/input/block.py", line 70, in __setattr__
      #   if hasattr(result, '__set__'): result.__set__(self, value)
      # File "keywords.py", line 1027, in __set__
      #   traceback.print_stack( file=sys.stdout)

      # Stack trace B, from pbsout:
      # File "ipython/launch/scattered_script.py", line 113, in <module>
      #   if __name__ == "__main__": main()
      # File "ipython/launch/scattered_script.py", line 108, in main
      #   jobfolder[name].compute(comm=comm, outdir=name)
      # File "jobfolder/jobfolder.py", line 297, in compute
      #   res = self.functional.__call__(**params)
      # File "<string>", line 16, in __call__
      # File "<string>", line 12, in iter
      # File "vasp/relax.py", line 277, in iter_relax
      #   **params
      # File "tools/__init__.py", line 46, in wrapper
      #   return function(self, structure, outdir, **kwargs)
      # File "tools/__init__.py", line 65, in wrapper
      #   if hasattr(self, key): setattr(self, key, kwargs.pop(key))
      # File "tools/input/block.py", line 70, in __setattr__
      #   if hasattr(result, '__set__'): result.__set__(self, value)
      # File "vasp/keywords.py", line 1027, in __set__
      #   traceback.print_stack( file=sys.stdout)


    if value is None: value = 'static'
    if hasattr(value, '__iter__'): value = ' '.join([str(u) for u in value])
    # try integer value
    try: dummy = int(value)
    except: pass
    else: 
      value = { 0: 'ionic', 1: 'ionic', 2: 'ionic', 3: 'cellshape ionic volume',
                4: 'cellshape ionic', 5: 'cellshape', 6: 'cellshape volume',
                7: 'volume' }[dummy]
    value = set(value.lower().replace(',', ' ').rstrip().lstrip().split())
    if bugLev >= 5:
      print 'keywords: Relaxation.set B: value: %s' % (value,)
      # Shows: set(['volume', 'cellshape', 'ionic'])
    result = []
    if 'all' in value:
      #result = 'ionic cellshape volume'.split()
      result = 'ionic cellshape volume gwcalc'.split() # gwmod
    else:
      if 'ion' in value or 'ions' in value or 'ionic' in value:
        result.append('ionic')
      if 'cell' in value or 'cellshape' in value or 'cell-shape' in value: 
        result.append('cellshape')
      if 'volume' in value: result.append('volume')
      if 'gwcalc' in value: result.append('gwcalc')   # gwmod
    if bugLev >= 5:
      print 'keywords: Relaxation.set C: result: %s' % (result,)
      # Shows: ['ionic', 'cellshape', 'volume']

    result = ', '.join(result)
    if bugLev >= 5:
      print 'keywords: Relaxation.set D: result: %s' % (result,)
      # Shows: ionic, cellshape, volume
      print 'keywords: Relaxation.set D: ibrion: %s' % (instance.ibrion,)
      print 'keywords: Relaxation.set D: isif: %s' % (instance.isif,)
      print 'keywords: Relaxation.set D: nsw: %s' % (instance.nsw,)
      # Shows: ibrion: None or 2
      # Shows: isif: None or 3
      # Shows: nsw: None or 50

    # static case
    if len(result) == 0:
    # gwmod?: if len(result) == 0 or result == ['gwcalc']:
      instance.nsw = 0
      if instance.ibrion is not None: instance.ibrion = -1
      if instance.isif is not None:
        if instance.isif > 2: instance.isif = 2
      return
    
    # non-static
    if instance.nsw is None or instance.nsw <= 0: instance.nsw = 50
    if instance.ibrion is None or instance.ibrion == -1: instance.ibrion = 2
    ionic = 'ionic' in result
    cellshape = 'cellshape' in result
    volume = 'volume' in result
    gwcalc = 'gwcalc' in result     #gwmod
    if bugLev >= 5:
      print 'keywords: Relaxation.set E: ionic: %s  cellshape: %s  volume: %s  gwcalc: %s' % (ionic, cellshape, volume, gwcalc,)
      # Shows: ionic: True  cellshape: True  volume: True  gwcalc: False

    instance.isif = 0    #gwmod
    if ionic and (not cellshape) and (not volume):   instance.isif = 2
    elif ionic and cellshape and (not volume):       instance.isif = 4
    elif ionic and cellshape and volume:             instance.isif = 3
    elif (not ionic) and cellshape and volume:       instance.isif = 6
    elif (not ionic) and cellshape and (not volume): instance.isif = 5
    elif (not ionic) and (not cellshape) and volume: instance.isif = 7
    elif ionic and (not cellshape) and volume: 
      raise ValueError( "VASP does not allow relaxation of atomic position "   \
                        "and volume at constant cell-shape.\n" )
    elif gwcalc:                                      instance.isif = 0
    else: instance.isif = 2

    # gwmod:
    if 'gwcalc' in value and instance.isif != 0:
      raise ValueError("cannot combine gw with other relaxation")

  def output_map(self, **kwargs): return None



class ISmear(AliasKeyword):
  keyword = 'ismear'
  aliases = { -5: ['metal'], -4: ['tetra'], -3: ['dynamic'],
              -1: ['fermi'], -2: ['fixresults'], 0: ['gaussian'],
               1: ['mp', 'mp1', 'mp 1'], 2: ['mp 2', 'mp2'],
               3: ['mp3', 'mp 3'] }


class Sigma(QuantityKeyword): 
  keyword  = 'sigma'
  units = eV


class LSorbit(BoolKeyword):
  """ Run calculation with spin-orbit coupling. 

      Accepts None, True, or False.
      If True, then sets :py:attr:`~pylada.vasp.incar.Incar.nonscf` to True.
      When printing INCAR stuff, checks for valid prior calculation. And sets
      lmaxmix to value of prior calculation.
  """ 
  keyword = 'lsorbit'
  """ VASP keyword """
  def __init__(self, value=None):
    super(LSorbit, self).__init__(value=value)
  def __get__(self, instance, owner=None): return self.value
  def __set__(self, instance, value):
    if value is None: self._value = None; return
    self.value = value == True
    if self.value:
      instance.ispin = 2
      instance.nonscf = True
  def output_map(self, **kwargs):
    if self.value is None or self.value == False: return None
    vasp = kwargs['vasp']
    if not vasp.nonscf:
      raise ValueError( 'Expected non-self-consistent '                        \
                        'calculation with LSORBIT = True' )
    if vasp.restart is None: 
      raise ValueError( 'Expected to restart from other '                      \
                        'calculation with LSORBIT = True' )
    if vasp.restart.success == False:
      raise ValueError( 'Self-consistent calculation was unsuccessful. '       \
                        'Cannot perform LSORBIT = True calculation.' )
    vasp.lmaxmix = vasp.restart.lmaxmix
    vasp.lvhar = vasp.restart.lvhar
    return super(LSorbit, self).output_map(**kwargs)


class NonScf(BoolKeyword):
  keyword = None
  """ Does not correspond to a VASP keyword. """
  def __init__(self, value=False):
    super(NonScf, self).__init__(value=False)


class LMaxMix(TypedKeyword):
  keyword = 'lmaxmix'
  type = int


class LVHar(BoolKeyword):
  keyword = 'LVHAR'
  def __init__(self, value=False):
    super(LVHar, self).__init__(value=value)
