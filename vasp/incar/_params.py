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

""" Standard parameter types for use as attributes in Incar """
__docformat__ = "restructuredtext en"
__all__ = [ "SpecialVaspParam", "ExtraElectron", "Algo", "Precision", "Ediff",\
            "Ediffg", "Encut", "EncutGW", "FFTGrid", "Restart", "UParams", "IniWave",\
            "Magmom", 'Npar', 'Boolean', 'Integer', 'Choices', 'PrecFock', 'NonScf', \
            "System", 'PartialRestart', 'Relaxation', 'Smearing', 'Lsorbit' ]
from quantities import eV
from pylada.misc import bugLev

class SpecialVaspParam(object): 
  """ Base type for special vasp parameters. 
  
      Special vasp parameters do something more than just print to the incar.
      What *more* means depends upon the parameter.

  """
  def __init__(self, value): 
    super(SpecialVaspParam, self).__init__()
    self.value = value
    """ Value derived classes will do something with. """
  def __repr__(self): return "{0.__class__.__name__}({1!r})".format(self, self.value)

class Magmom(SpecialVaspParam):
  """ Sets the initial magnetic moments on each atom.

      There are three types of usage: 
	- do nothing if the instance's value is None or False, or if not a
	  single atom has a 'magmom' attribute
        - print a string preceded by "MAGMOM = " if the instance's value is a string. 
        - print the actual MAGMOM string from the magnetic moments attributes
          ``magmom`` in the structure's atoms if anything but a string, None,
          or False.

      If the calculation is **not** spin-polarized, then the magnetic moment
      tag is not set.

      .. seealso:: `MAGMOM <http://cms.mpi.univie.ac.at/vasp/guide/node100.html>`_
  """
  def __init__(self, value=True):
    super(Magmom, self).__init__(value)
    
  def incar_string(self, **kwargs):
    from ...crystal import specieset
    if self.value is None or self.value == False: return None
    if kwargs["vasp"].ispin == 1: return None
    if isinstance(self.value, str): return "MAGMOM = {0}".format(self.value)
    
    structure = kwargs['structure']
    if all(not hasattr(u, 'magmom') for u in structure): return None
    result = ""
    for specie in specieset(structure):
      moments = [getattr(u, 'magmom', 0e0) for u in structure if u.type == specie]
      tupled = [[1, moments[0]]]
      for m in moments[1:]: 
        # Change precision from 1.e-12 to 1.e-1, per Vladan, 2013-10-09
        #if abs(m - tupled[-1][1]) < 1e-12: tupled[-1][0] += 1
        if abs(m - tupled[-1][1]) < 1e-1: tupled[-1][0] += 1
        else: tupled.append([1, m])
      for i, m in tupled:
        # Change format from .2f to .1f, per Vladan, 2013-10-09
        if i == 1: result += "{0:.1f} ".format(m)
        else:      result += "{0}*{1:.1f} ".format(i, m)
    return 'MAGMOM = {0}'.format(result.rstrip())
  
class System(SpecialVaspParam):
  """ System title to use for calculation.

      Adds system name to OUTCAR. If value is the python object ``True``, the
      structure is checked for a ``name`` attribute.  If it is False or None,
      SYSTEM is not added to the incar. In all other case, tries to convert the
      result to a string and use that.

      The call is protected by a try statement.

      .. seealso:: `SYSTEM <http://cms.mpi.univie.ac.at/vasp/guide/node94.html>`_
  """

  def __init__(self, value): super(System, self).__init__(value)

  def incar_string(self, **kwargs):
    if self.value is None or self.value is False: return None
    try: 
      if self.value is True: 
        if not hasattr(kwargs["structure"], "name"): return None
        name = kwargs["structure"].name.rstrip().lstrip()
        if len(name) == 0: return None
        return "SYSTEM = {0}".format(name)
      return "SYSTEM = {0}".format(self.value)
    except: return None
    
class Npar(SpecialVaspParam):
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

  def __init__(self, value): super(Npar, self).__init__(value)

  def incar_string(self, **kwargs):
    from math import log, sqrt
    if self.value is None: return None
    if not isinstance(self.value, str): 
      if self.value < 1: return None
      return "NPAR = {0}".format(self.value)

    comm = kwargs.get('comm', None)
    n = getattr(comm, 'n', getattr(comm, 'size', -1))
    if n == 1: return None
    if self.value == "power of two":
      m = int(log(n)/log(2))
      for i in range(m, 0, -1):
        if n % 2**i == 0: return "NPAR = {0}".format(i)
      return None
    if self.value == "sqrt": return "NPAR = {0}".format(int(sqrt(n)+0.001))
    raise ValueError("Unknown request npar = {0}".format(self.value))
    
class ExtraElectron(SpecialVaspParam):
  """ Sets number of electrons relative to neutral system.
      
      Gets the number of electrons in the (neutral) system. Then adds value to
      it and computes with the resulting number of electrons.

      >>> vasp.extraelectron =  0  # charge neutral system
      >>> vasp.extraelectron =  1  # charge -1 (1 extra electron)
      >>> vasp.extraelectron = -1  # charge +1 (1 extra hole)

      :param integer value:
        Number of electrons to add to charge neutral system. Defaults to 0.

      .. seealso:: `NELECT <http://cms.mpi.univie.ac.at/vasp/vasp/NELECT.html>`_
  """

  def __init__(self, value=0): super(ExtraElectron, self).__init__(value)

  def nelectrons(self, vasp, structure):
    """ Total number of electrons in the system """
    from math import fsum
    # constructs dictionnary of valence charge
    valence = {}
    for key, value in vasp.species.items():
      valence[key] = value.valence
    # sums up charge.
    return fsum( valence[atom.type] for atom in structure )
    
  def incar_string(self, **kwargs):
    # gets number of electrons.
    charge_neutral = self.nelectrons(kwargs['vasp'], kwargs['structure'])
    # then prints incar string.
    if self.value == 0:
      return "# NELECT = {0} Charge neutral system".format(charge_neutral)
    elif self.value > 0:
      return "NELECT = {0}  # negatively charged system ({1})"\
             .format(charge_neutral + self.value, -self.value)
    else: 
      return "NELECT = {0}  # positively charged system (+{1})"\
             .format (charge_neutral + self.value, -self.value)
          
      
class Algo(SpecialVaspParam): 
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
  def __init__(self, value="fast"): super(Algo, self).__init__(value)
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
    if is_vasp_4                                         \
       and ( lower[0] in ['c', 's', 'e']                 \
             or lower in [ "nothing", "subrot", "exact", \
                           "gw", "gw0", "chi", "scgw",   \
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
    
  def incar_string(self, **kwargs):
    if self.value is None: return None
    return "ALGO = {0}".format(self.value)

class Ediff(SpecialVaspParam):
  """ Sets the convergence criteria (per atom) for electronic minimization.

      - value > 0e0: the tolerance is multiplied by the number of atoms in the
        system. This makes tolerance consistent from one system to the next.
      - value < 0e0: tolerance is given as absolute value, without multiplying
        by size of system.

      .. seealso:: `EDIFF <http://cms.mpi.univie.ac.at/vasp/guide/node105.html>`_
  """
  def __init__(self, value):
    """ Creates *per atom* tolerance. """
    super(Ediff, self).__init__(value)
    print "vasp/incar/_params: Ediff.const: value: %s" % (value,)
  def incar_string(self, **kwargs):
    if self.value is None: return 
    if self.value < 0: return "EDIFF = {0} ".format(-self.value)
    res = "EDIFF = {0} ".format(self.value * float(len(kwargs["structure"])))
    print "vasp/incar/_params: Ediff.incar_string: res: %s" % (res,)
    return res
  def __repr__(self):
    return "{0.__class__.__name__}({0.value!r})".format(self)

class Ediffg(SpecialVaspParam):
  """ Sets the convergence criteria (per atom) for ionic minimization.

      - value > 0e0: the tolerance is multiplied by the number of atoms in the
        system. This makes tolerance consistent from one system to the next.
      - value < 0e0: tolerance is given as is (negative), and applies to forces.

      .. seealso:: `EDIFFG <http://cms.mpi.univie.ac.at/vasp/guide/node107.html>`_
  """
  def __init__(self, value):
    """ Creates *per atom* tolerance. """
    super(Ediffg, self).__init__(value)
  def incar_string(self, **kwargs):
    if self.value is None: return 
    if self.value < 0: return "EDIFFG = {0} ".format(self.value)
    return "EDIFFG = {0} ".format(self.value * float(len(kwargs["structure"])))
  def __repr__(self):
    return "{0.__class__.__name__}({0.value!r})".format(self)

class Encut(SpecialVaspParam):
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
  KEY = "ENCUT"
  """ Corresponding VASP key. """
  units = eV
  """ Units with which to sign cutoff. """
  def __init__(self, value): super(Encut, self).__init__(value)
  @property
  def value(self):
    """ Returns value signed by a physical unit. """
    if self._value is None: return None
    if self._value <= 1e-12: return None
    return self._value if self._value <= 3.0 else self._value * self.units
  @value.setter
  def value(self, value):
    """ Sets value taking unit into account. """
    if hasattr(value, 'rescale'): value = value.rescale(self.units).magnitude
    self._value = value

  def incar_string(self, **kwargs):
    from ...crystal import specieset
    value = self._value
    if value is None:   return None
    elif value < 1e-12: return None
    elif value >= 1e-12 and value <= 3.0:
      types = specieset(kwargs["structure"])
      encut = max(kwargs["vasp"].species[type].enmax for type in types)
      if hasattr(encut, 'rescale'): encut = float(encut.rescale(eV))
      return "{0} = {1} ".format(self.KEY, encut * value)
    return "{0} = {1}".format(self.KEY, value)

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
  KEY = "ENCUTGW"
  def __init__(self, value): super(EncutGW, self).__init__(value)

class FFTGrid(SpecialVaspParam):
  """ FFT mesh of the wavefunctions.
  
      This must a sequence of three integers.

      .. seealso:: `NGX, NGY, NGZ
        <http://cms.mpi.univie.ac.at/vasp/guide/node93.html>`_
  """
  def __init__(self, value): super(FFTGrid, self).__init__(value)
  @property 
  def value(self): return self._value
  @value.setter
  def value(self, value): 
    from numpy import array
    if value is None:
      self._value = None
      return
    if len(list(value)) != 3: raise TypeError("FFTGrid expects three numbers.")
    self._value = array(value)
  def incar_string(self, **kwargs):
    if self.value is None: return None
    return "NGX = {0[0]}\nNGY = {0[1]}\nNGZ = {0[2]}".format(self.value)

class PartialRestart(SpecialVaspParam):
  """ Restart from previous run.
      
      It is either an vasp extraction object of some kind, or None.  In the
      latter case, the calculation starts from scratch.  However, if an
      extraction object exists *and* the calculation it refers to was
      successfull, then it will check whether WAVECAR and CHGCAR exist and set
      ISTART_ and ICHARG_ accordingly. It also checks whether
      :py:attr:`nonscf <incar.Incar.nonscf>` is True or False, and sets
      ICHARG_ accordingly. The CONTCAR file is *never* copied from the
      previous run. For an alternate behavior, see :py:class:`Restart`.

      .. seealso:: ICHARG_, ISTART_, :py:class:`Restart`
  """
  def __init__(self, value): super(PartialRestart, self).__init__(value)

  def incar_string(self, **kwargs):
    from os.path import join, exists, getsize
    from shutil import copy
    from ...misc import copyfile
    from .. import files

    if self.value is None or self.value.success == False:
      if kwargs['vasp'].nonscf: kwargs['vasp'].icharg = 12
      if bugLev >= 5: print 'vasp/incar/_params: PartialRestart: no luck'
      return None
    else:
      if bugLev >= 5:
        print 'vasp/incar/_params: PartialRestart: self.val.dir: %s' \
          % (self.value.directory,)
      ewave = exists( join(self.value.directory, files.WAVECAR) )
      if ewave: ewave = getsize(join(self.value.directory, files.WAVECAR)) > 0
      if ewave:
        copy(join(self.value.directory, files.WAVECAR), ".")
        kwargs['vasp'].istart = 1
      else: kwargs['vasp'].istart = 0
      echarg = exists( join(self.value.directory, files.CHGCAR) )
      if echarg: echarg = getsize(join(self.value.directory, files.CHGCAR)) > 0
      if echarg:
        copy(join(self.value.directory, files.CHGCAR), ".")
        kwargs['vasp'].icharg = 1
      else: kwargs['vasp'].icharg = 0 if kwargs['vasp'].istart == 1 else 2
      if getattr(kwargs["vasp"], 'nonscf', False): kwargs['vasp'].icharg += 10

      copyfile(join(self.value.directory, files.EIGENVALUES), nothrow='same exists',
               nocopyempty=True) 
      copyfile(join(self.value.directory, files.WAVEDER), files.WAVEDER,
               nothrow='same exists', symlink=getattr(kwargs["vasp"], 'symlink', False),
               nocopyempty=True) 
      copyfile(join(self.value.directory, files.TMPCAR), files.TMPCAR,
               nothrow='same exists', symlink=getattr(kwargs["vasp"], 'symlink', False),
               nocopyempty=True) 
      if kwargs['vasp'].lsorbit == True: kwargs['vasp'].nbands = 2*self.value.nbands 
    return None

class Restart(PartialRestart):
  """ Return from previous run from which to restart.
      
      Restart from a previous run, as described in :py:class:`PartialRestart`.
      However, unlike :py:class:`PartialRestart`, the CONTCAR is copied from
      the previous run, if it exists and is not empty.

      .. seealso:: ICHARG_, ISTART_
  """
  def __init__(self, value): super(Restart, self).__init__(value)

  def incar_string(self, **kwargs):
    from os.path import join
    from ...misc import copyfile
    from .. import files
    import os

    result = super(Restart, self).incar_string(**kwargs)
    if self.value is not None and self.value.success:
      copyfile(join(self.value.directory, files.CONTCAR), files.POSCAR,\
               nothrow='same exists', symlink=getattr(kwargs["vasp"], 'symlink', False),\
               nocopyempty=True) 
    if bugLev >= 5:
      print 'vasp/incar/_params: Restart CONTCAR: self.val.dir: %s' \
        % (self.value.directory,)
      print 'vasp/incar/_params: Restart: os.getcwd():  %s' \
        % (os.getcwd(),)
      print 'vasp/incar/_params: Restart: result: %s' % (result,)
    return result

class NonScf(SpecialVaspParam):
  """ Whether to perform a self-consistent or non-self-consistent run. 
  
      Accepts only True or False(default). This parameter works with
      :py:class:`Restart` to determine the value to give :py:attr:`icharg
      <pylada.vasp.incar.Incar.icharg>`
  """
  def __init__(self, value):  super(NonScf, self).__init__(value)
  @property
  def value(self): return self._value
  @value.setter
  def value(self, value):
    if isinstance(value, str):
      if len(value) == 0: value = False
      elif   value.lower() == "true"[:min(len(value), len("true"))]: value = True
      elif value.lower() == "false"[:min(len(value), len("false"))]: value = False
      else: raise RuntimeError("Uknown value for nonscf: {0}").format(value)
    self._value = value == True

  def incar_string(self, **kwargs): return None
  def __repr__(self): return "{0.__class__.__name__}({0.value!r})".format(self)

class UParams(SpecialVaspParam): 
  """ Sets U, nlep, and enlep parameters. 
 
      The U, nlep, and enlep parameters of the atomic species are set at the
      same time as the pseudo-potentials. This object merely sets up the incar
      with right input.

      However, it does accept one parameter, which can be "off", "on", "occ" or
      "all" wich defines the level of verbosity of VASP (with respect to the
      parameters).


      .. seealso:: `LDAU, LDAUTYPE, LDAUL, LDAUPRINT
        <http://cms.mpi.univie.ac.at/vasp/vasp/On_site_Coulomb_interaction_L_S_DA_U.html>`_
  """
  def __init__(self, value):
    import re
    
    if value is None: value = 0
    elif hasattr(value, "lower"): 
      value = value.lower() 
      if value == "off": value = 0
      elif value == "on": value = 1
      elif None != re.match(r"\s*occ(upancy)?\s*", value): value = 1
      elif None != re.match(r"\s*(all|pot(ential)?)\s*", value): value = 2

    super(UParams, self).__init__(value)

  def incar_string(self, **kwargs):
    from ...crystal import specieset
    if bugLev >= 5: print 'vasp/incar/_params: UParams.incar_string:'
    types = specieset(kwargs['structure'])
    species = kwargs['vasp'].species
    # existence and sanity check
    has_U, which_type = False, None 
    for type in types:
      specie = species[type]
      if bugLev >= 5:
        print '    check: type: %s  specie: %s  specie.U: %s  len: %s' \
          % (type, specie, specie.U, len(specie.U),)
      if len(specie.U) == 0: continue
      if len(specie.U) > 4: 
        raise AssertionError, "More than 4 channels for U/NLEP parameters"
      has_U = True
      # checks consistency.
      which_type = specie.U[0]["type"]
      if bugLev >= 5: print '    check: which_type: %s' % (which_type,)
      for l in specie.U[1:]: 
        assert which_type == l["type"], \
               AssertionError("LDA+U/NLEP types are not consistent across species.")
    if not has_U: return "# no LDA+U/NLEP parameters";

    # Prints LDA + U parameters
    result = "LDAU = .TRUE.\nLDAUPRINT = {0}\nLDAUTYPE = {1}\n".format(self.value, which_type)
    if bugLev >= 5:
      print '    self.value: %s' % (self.value,)
      print '    which_type: %s' % (which_type,)
      print '    result: %s' % (result,)

    for i in range( max(len(species[type].U) for type in types) ):
      line = "LDUL{0}=".format(i+1), "LDUU{0}=".format(i+1), "LDUJ{0}=".format(i+1), "LDUO{0}=".format(i+1)
      if bugLev >= 5: print '      i: %s  line: %s' % (i, line,)
      for type in types:
        specie = species[type]
        a = -1, 0e0, 0e0, 1
        if bugLev >= 5:
          print '        type: %s  specie: %s  len: %d' \
            % (type, specie, len(specie.U),)
        if len(specie.U) <= i: pass
        else:
          if bugLev >= 5:
            print '          func: %s' % (specie.U[i]["func"],)
            print '          l: %s' % (specie.U[i]["l"],)
            print '          U: %s' % (specie.U[i]["U"],)
            print '          J: %s' % (specie.U[i]["J"],)
            print '          U0: %s' % (specie.U[i]["U0"],)
            print '          U1: %s' % (specie.U[i]["U1"],)
          if specie.U[i]["func"] == "U":    
            a = [specie.U[i]["l"], specie.U[i]["U"], specie.U[i]["J"], 1]
          elif specie.U[i]["func"] == "nlep": 
            a = [specie.U[i]["l"], specie.U[i]["U0"], 0e0, 2]
          elif specie.U[i]["func"] == "enlep":
            a = [specie.U[i]["l"], specie.U[i]["U0"], specie.U[i]["U1"], 3]
          else: raise RuntimeError, "Debug Error."
        if bugLev >= 5: print '        a: %s' % (a,)
        if hasattr(a[1], "rescale"): a[1] = a[1].rescale("eV")
        if hasattr(a[2], "rescale"): a[2] = a[2].rescale("eV")
        line = "{0[0]} {1[0]}".        format(line, a),\
               "{0[1]} {1[1]:18.10e}". format(line, a),\
               "{0[2]} {1[2]:18.10e}".format(line, a),\
               "{0[3]} {1[3]}".        format(line, a)
        if bugLev >= 5: print '        line: %s' % (line,)
      result += "\n{0}\n{1}\n{2}\n{3}\n".format(*line)
      if bugLev >= 5: print '      result: %s' % (result,)
    if bugLev >= 5: print '    final result: %s' % (result,)
    return result
  def __repr__(self):
    return "{0.__class__.__name__}({1!r})".format(self, ["off", "on", "all"][self.value])

class Boolean(SpecialVaspParam):
  """ Any boolean vasp parameters. 
  
      Python is very liberal in how it converts any object to a boolean, eg an
      empty dictionary is false while non-empty dictionary is true.  In order
      to keep this behavior, the value given to this parameter is kept as is as
      long as possible, and converted only when writing the incar. The only
      difference with the python behavior is that if using strings (which
      generally evaluate to true or depending whether or not they are empty),
      these must be "True" or "False", or variations thereoff. 'on' and 'off'
      evaluate to True and False, respectively. The empty string will
      evaluate to the VASP default (eg equivalent to using None). 
  """
  def __init__(self, key, value):
    super(Boolean, self).__init__(value)
    self.key = key
    """ VASP key corresponding to this input. """
  @property
  def value(self):
    return self._value
  @value.setter
  def value(self, value):
    if value is None: self._value = None; return
    elif isinstance(value, str):
      value = value.lstrip().rstrip.lower()
      if len(value) == 0: return
      elif value == "on": value = True
      elif value == "off": value = False
      elif value == "true"[:min(len(value), len("true"))]: value = True
      elif value == "false"[:min(len(value), len("false"))]: value = False
      else: raise TypeError("Cannot interpret string {0} as a boolean.".format(value))
    self._value = value == True
  def incar_string(self, **kwargs):
    value = self._value
    if isinstance(value, str):
      if len(value) == 0: value is None 
      elif value.lower() == "true"[:len(value)]: value = True
      else: value = False
    if self.value is None: return None
    return "{0} = {1}".format(self.key.upper(), ".TRUE." if bool(self.value) else ".FALSE.")
  def __repr__(self):
    """ Representation of this object. """
    return "{0.__class__.__name__}({1!r}, {2!r})".format(self, self.key, self.value)

class Integer(SpecialVaspParam):
  """ Any integer vasp parameters. 
  
      The value is always of type integer. Other types are converted to an
      integer where possible, and will throw TypeError otherwise.
  """
  def __init__(self, key, value):
    super(Integer, self).__init__(value)
    self.key = key
    """ VASP key corresponding to this input. """
  @property 
  def value(self): return self._value
  @value.setter
  def value(self, value):
    if value is None: self._value = None; return
    try: self._value = int(value)
    except: raise TypeError("Could not evaluate {0} as an integer.".format(value))
  def incar_string(self, **kwargs):
    if self.value is None: return None
    return "{0} = {1}".format(self.key.upper(), self.value)
  def __repr__(self):
    """ Representation of this object. """
    return "{0.__class__.__name__}({1}, {2})".format(self, repr(self.key), repr(self.value))

class Choices(SpecialVaspParam):
  """ Vasp parameters with a limited set of choices. 

      Initializes the Choices-type vasp parameters.

      :param key:
          Name of the VASP parameter, e.g. "precfock". It needs not be in
          uppercase. In fact, lower case is preferred for being more pythonic.
      :param choices:
          Dictionary where key is an allowed VASP input for this parameter.
          To each key is associated a list (or set), with allowable forms
          which will translate to the key in the incar. A modified copy of
          this dictionary is owned by the instance being initialized. All
          keys and items should be meaningfully convertible to strings.
      :param default:
          Option from ``choices`` to use as default.

      .. note:: The keys are case-sensitive. The values are not.
  """
  def __init__(self, key, choices, default=None):
    self.key = key
    """ VASP key corresponding to this input. """
    self.choices = {}
    """ Allowable set of choices. """
    for key, items in choices.iteritems():
      self.choices[key] = [u.lower() if hasattr(u, 'lower') else u for u in items]
      self.choices[key].append(key.lower() if hasattr(key, 'lower') else key)
    super(Choices, self).__init__(default)

  @property
  def value(self): return self._value
  @value.setter
  def value(self, value):
    if value is None: self._value = None; return
    if hasattr(value, 'lower'): value = value.lower()
    for key, items in self.choices.iteritems():
      if value in items: self._value = key; return
    raise ValueError("{0} is not an acceptable choice for {1.key}: {1.choices}.".format(value, self))
  def incar_string(self, **kwargs):
    if self.value is None: return None
    return "{0} = {1}".format(self.key.upper(), self.value)
  def __repr__(self):
    """ Representation of this object. """
    return "{0.__class__.__name__}({1}, {2}, {3})"\
           .format(self, repr(self.key), repr(self.choices), repr(self.value))

class PrecFock(Choices):
  """ Sets up FFT grid in hartree-fock related routines.
      
      Allowable options are:

      - low
      - medium
      - fast
      - normal
      - accurate

      .. note:: The values are not case-sensitive. 
      .. seealso:: `PRECFOCK  <http://cms.mpi.univie.ac.at/vasp/vasp/PRECFOCK_FFT_grid_in_HF_related_routines.html>`_
  """
  def __init__(self, value=None):
    choices = { 'Low': ['low'], 'Medium': ['medium'], 'Fast': ['fast'],
                'Normal': ['normal'], 'Accurate': ['accurate'] }
    super(PrecFock, self).__init__("PRECFOCK", choices, value)
  def __repr__(self):
    return "{0.__class__.__name__}({0.value!r})".format(self)

class Precision(Choices):
  """ Sets accuracy of calculation. 

      - accurate (default)
      - low
      - medium
      - high
      - single

      .. seealso:: `PREC <http://cms.mpi.univie.ac.at/vasp/vasp/PREC_tag.html>`_
  """
  def __init__(self, value = 'accurate'):
    choices = { 'Accurate': ['accurate'], 'Low': ['low'], 'Normal': ['normal'],
                'Medium': ['medium'], 'High': ['high'], 'Single': ['single'] }
    super(Precision, self).__init__('PREC', choices, value)
  def __repr__(self):
    return "{0.__class__.__name__}({0.value!r})".format(self)

class IniWave(Choices):
  """ Specifies how to setup initial wavefunctions.
  
      - 0, jellium
      - 1, random 

      .. seealso:: `INIWAV <http://cms.mpi.univie.ac.at/vasp/guide/node103.html>`_
  """
  def __init__(self, value=None):
    choices = {0: ['jellium'], 1: ['random']}
    super(IniWave, self).__init__('INIWAV', choices, value)
  def __repr__(self):
    return "{0.__class__.__name__}({1!r})".format(self, self.choices[self.value][0])

class Relaxation(SpecialVaspParam):
  """ Sets type of relaxation.
  
      Defaults to None, eg use VASP defaults for ISIF_, NSW_, IBRION_, POTIM_.
      It can be set to a single value, or to a tuple of up to four elements:

      >>> vasp.relaxation = "static" 
      >>> vasp.relaxation = "static", 20
    
      - first argument can be "static", or a combination of "ionic",
        "cellshape", and "volume". The combination must be allowed by
        ISIF_. It can also be an integer, in which case ISIF_ is set
        directly.
      - second (optional) argument is NSW_
      - third (optional) argument is IBRION_
      - fourth (optional) argument is POTIM_

      .. warning: When the first parameter is one of "cellshape", "volume",
         "ionic"  and yet the second (NSW_) is None, 0 or not present, then NSW_
         is set to 50.  The assumption is when you ask to relax, then indeed
         you do ask to relax. 
      .. warning:  When the first parameter is one of "cellshape", "volume",
         "ionic" and IBRION_ is not specified or None, then IBRION_ is to 2. 
         In contrast, VASP defaults IBRION_ to zero. However, that's just
         painful.
      
      .. _ISIF: http://cms.mpi.univie.ac.at/vasp/guide/node112.html
      .. _NSW: http://cms.mpi.univie.ac.at/vasp/guide/node108.html
      .. _IBRION: http://cms.mpi.univie.ac.at/vasp/guide/node110.html
      .. _POTIM: http://cms.mpi.univie.ac.at/vasp/vasp/POTIM_tag.html
  """
  def __init__(self, isif=None, nsw=None, ibrion=None, potim=None): 
    super(Relaxation, self).__init__((isif, nsw, ibrion, potim))

  @property
  def value(self):
    if self.isif is None and self.ibrion is None and self.potim is None and self.nsw is None: 
      return None
    result = [None, self.nsw, self.ibrion, self.potim]
    if self.ibrion == -1 or self.nsw == 0:
      result[0] = 'static'
      result[2] = None
    elif self.isif is not None and self.ibrion != -1:
      if result[0] is None: result[0] = ''
      if self.isif < 5: result[0] += ' ionic'
      if self.isif > 2 and self.isif < 7: result[0] += ' cellshape'
      if self.isif in [3, 6, 7]: result[0] += ' volume'
      result[0] = result[0].lstrip()
      if self.nsw == 50 and self.ibrion is None and self.potim is None: result[1] = None
    for i in xrange(4): 
      if result[-1] is None: result = result[:-1]
    if len(result) == 1: return result[0]
    if len(result) == 3 and result[0] != 'static' and result[2] == 2:
      return tuple(result[:2]) if result[1] != 50 else result[0]
    return tuple(result)

  @value.setter
  def value(self, args): 
    import re

    if 'nsw'    not in self.__dict__: self.nsw    = None
    if 'isif'   not in self.__dict__: self.isif   = None
    if 'ibrion' not in self.__dict__: self.ibrion = None
    if 'potim'  not in self.__dict__: self.potim  = None
    if args == None: 
      self.isif = None
      return

    isif, nsw, ibrion, potim = None, None, None, None
    if hasattr(args, 'lower'): dof = args.lower().rstrip().lstrip()
    elif hasattr(args, '__len__') and hasattr(args, '__getitem__'):
      if len(args) > 0:
        if hasattr(args[0], 'lower'): dof = args[0].lower().rstrip().lstrip()
        elif args[0] is None: isif, dof = None, None
        else: isif, dof = int(args[0]), None
      if len(args) > 1 and args[1] is not None: nsw    = int(args[1])
      if len(args) > 2 and args[2] is not None: ibrion = int(args[2])
      if len(args) > 3 and args[3] is not None: potim  = float(args[3])
    else: isif, dof = int(args), None

    if dof is not None:
      if dof == 'all': dof = 'ionic cellshape volume'
      ionic = re.search( "ion(ic|s)?", dof ) is not None
      cellshape = re.search( "cell(\s+|-|_)?(?:shape)?", dof ) is not None
      volume = re.search( "volume", dof ) is not None
      
      # static calculation.
      if (not ionic) and (not cellshape) and (not volume):
        if dof != 'static':
          raise RuntimeError("Unkown value for relaxation: {0}.".format(dof))
        isif = 2
        ibrion = -1
      else: # Some kind of relaxations. 
        # ionic calculation.
        if ionic and (not cellshape) and (not volume):   isif = 2
        elif ionic and cellshape and (not volume):       isif = 4
        elif ionic and cellshape and volume:             isif = 3
        elif (not ionic) and cellshape and volume:       isif = 6
        elif (not ionic) and cellshape and (not volume): isif = 5
        elif (not ionic) and (not cellshape) and volume: isif = 7
        elif ionic and (not cellshape) and volume: 
          raise RuntimeError, "VASP does not allow relaxation of atomic position "\
                              "and volume at constant cell-shape.\n"
        if nsw == 0: 
          raise ValueError("Cannot set nsw < 1 and perform strain relaxations.")
        elif nsw is None: nsw = 50
    if isif is None and dof is not None: 
      raise ValueError("Unexpected argument to relaxation: {0}.".format(dof))
    if nsw    is not None: self.nsw    = nsw
    if isif   is not None: self.isif   = isif
    if ibrion is not None: self.ibrion = ibrion
    if potim  is not None: self.potim  = potim
    if self.ibrion is None and self.nsw is not None and self.nsw > 1 and self.potim is None: 
      self.ibrion = 2

  def incar_string(self, **kwargs):
    if self.value is None: return None
    result = "ISIF = {0}\n".format(self.isif) if self.isif is not None else ''
    if self.nsw != None and self.ibrion != -1 and self.nsw != 0:
      result += "NSW = {0}\n".format(self.nsw)
    if self.potim != None and self.ibrion != -1 and self.nsw != 0:
      result += "POTIM = {0}\n".format(self.potim)
    if self.ibrion != None: result += "IBRION = {0}\n".format(self.ibrion)
    vasp = kwargs['vasp']
    structure = kwargs['structure']
    if self.ibrion != -1 and vasp.ediffg is not None:
      if vasp.ediffg < vasp.ediff and vasp.ediffg > 0 and vasp.ediff > 0: 
        raise RuntimeError("Using ediffg (positive) smaller than ediff does not make sense.")
      if vasp.ediffg > 0 and vasp.ediff < 0 and abs(vasp.ediff) > vasp.ediffg * float(len(structure)): 
        raise RuntimeError("Using ediffg (positive) smaller than ediff does not make sense.")
    if result[-1] == '\n': result = result[:-1]
    return result

  def __repr__(self):
    value = self.value
    if value is None: return "{0.__class__.__name__}(None)".format(self)
    if isinstance(value, str):  return "{0.__class__.__name__}({1!r})".format(self, value)
    return "{0.__class__.__name__}({1})".format(self, repr(self.value)[1:-1])

class Smearing(SpecialVaspParam):
  """ Value of the smearing used in the calculation. 
  
      It can be specified as a string:
        
      >>> vasp.smearing = "type", x
     
      Where type is any of "fermi", "gaussian", "mp N", "tetra",
      "metal", or "insulator", and x is the energy scale.

      - fermi: use a Fermi-Dirac broadening.
      - gaussian: uses a gaussian smearing.
      - mp N: is for Methfessel-Paxton, where N is an integer indicating the
        order the mp method.
      - tetra: tetrahedron method without Bloechl correction.
      - bloechl: means tetrahedron method with Bloechl correction.
      - metal: equivalent to "mp 1 x"
      - insulator: is equivalent to "bloechl".
      - dynamic: corresponds to ISMEAR=-3.

      .. seealso:: `ISMEAR, SIGMA <http://cms.mpi.univie.ac.at/vasp/guide/node124.html>`_
  """
  def __init__(self, type=None, sigma=None):
    super(Smearing, self).__init__((type, sigma))

  @property 
  def value(self):
    if self.ismear is None and self.sigma is None: return None
    ismear = { -1: 'fermi', 0: 'gaussian', 1: 'metal', -5: 'insulator', -3: 'dynamic',
               -4: 'tetra', 2: 'mp 2', 3: 'mp 3', None: None}[self.ismear]
    if self.sigma is None: return ismear
    sigma = self.sigma
    if not hasattr(sigma, 'rescale'): sigma *= eV
    return ismear, sigma

  @value.setter
  def value(self, args):
    if args is None: 
      self.ismear, self.sigma = None, None
      return

    if isinstance(args, str): ismear, sigma = args, None
    elif len(args) == 1: ismear, sigma = args[0], None
    elif len(args) == 2: ismear, sigma = args
    else: raise ValueError("Incorrect input to smearing: {0}.".format(args))

    if hasattr(ismear, 'lower'): 
      ismear = ismear.rstrip().lstrip().replace(' ', '').lower()
      ismear = { 'fermi': -1, 'gaussian': 0, 'metal': 1, 'bloechl': -5, 'dynamic': -3, \
                 'tetra': -4, 'insulator': -5, 'mp1': 1, 'mp2': 2, 'mp3': 3, None: None }[ismear]
    elif ismear is not None: 
      ismear = int(ismear)
      if ismear < -5 or ismear > 3: raise RuntimeError("Unknown value for ismear: {0}.\n".format(ismear))
    self.ismear = ismear
    if sigma is not None:
      self.sigma = sigma
      if not hasattr(self.sigma, 'rescale'): self.sigma *= eV
      else: self.sigma = self.sigma.rescale(eV)
      if len(self.sigma.shape) > 0 and self.ismear is not None and self.ismear != -3:
        raise RuntimeError('Cannot use more than one smearing '\
                           'parameter with ismear={0}.'.format(self.ismear))
    elif len(args) == 2: self.sigma = None

  def incar_string(self, **kwargs):
    result = ''
    if self.ismear is not None:
      result = 'ISMEAR = {0}\n'.format(self.ismear)
    if self.sigma is not None:
      sigma = self.sigma.rescale(eV).magnitude if hasattr(self.sigma, 'rescale')\
              else self.sigma
      if len(sigma.shape) == 0:
        result += 'SIGMA = {0}'.format(sigma)
      else: 
        if self.ismear is None: result += 'ISMEAR = -3\n'
        result += 'SIGMA ='
        for u in sigma: result += ' {0}'.format(u)
    if len(result) == 0: return None
    return result[:-1] if result[-1] == '\n' else result

  def __repr__(self):
    value = self.value
    if value is None: return "{0.__class__.__name__}(None)".format(self)
    if isinstance(value, str):  return "{0.__class__.__name__}({1!r})".format(self, value)
    return "{0.__class__.__name__}({1})".format(self, repr(self.value)[1:-1])


class Lsorbit(Boolean): 
  """ Run calculation with spin-orbit coupling. 

      Accepts None, True, or False.
      If True, then sets :py:attr:`~pylada.vasp.incar.Incar.nonscf` to True and
      :py:attr:`~pylada.vasp.incar.Incar.ispin` to 2.
  """ 
  def __init__(self, value=None):
    super(Lsorbit, self).__init__('LSORBIT', value)
  def incar_string(self, **kwargs):
    if self.value == True:
      kwargs['vasp'].nonscf = True
      kwargs['vasp'].ispin  = 2
    return super(Lsorbit, self).incar_string(**kwargs)
