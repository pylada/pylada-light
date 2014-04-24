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

def U(type=1, l=2, U=0e0, J=0e0 ):
  """ Creates an LDA+U parameter 

      LDA+U is always LSDA+U here. 

      :param type:
          Should be one of the following: 1|2|"liechtenstein"|"dudarev".
          Specifies the type of the Hubbard U. Defaults to 1.
      :param l:
          Should be one of the following: 0|1|2|3|"s"|"p"|"d"|"f".
          Channel for which to apply U. Defaults to 2.
      :param U: 
          Hubbard U. Defaults to 0.
      :param J: float
          Hubbard J. Defaults to 0.

      :returns: A dictionary equivalent to the input.
  """
  if hasattr(type, "lower"):
    type = type.lower()
    if type == "liechtenstein": type = 1
    elif type == "dudarev": type = 2
    else: raise ValueError("Unknown value for type: {0}.".format(type))
  if hasattr(l, "lower"):
    l = l.lower()
    if len(l) != 1: raise ValueError("Uknown input {0}.".format(l))
    if   l[0] == 's': l = 0
    elif l[0] == 'p': l = 1
    elif l[0] == 'd': l = 2
    elif l[0] == 'f': l = 3
    else: raise ValueError("Uknown input {0}.".format(l)) 
  try: l = int(l)
  except: raise ValueError, "Moment l should be 0|1|2|3|s|p|d|f." 
  if l < 0 or l > 3: raise ValueError("Moment l should be 0|1|2|3|s|p|d|f.")
  if type != 1 and type != 2: raise ValueError("Unknown LDA+U type: {0}.".format(type))
  return { "type": int(type), "l": l, "U": U, "J": J, "func": "U" }


def nlep(type = 1, l=2, U0=0e0, U1=None, fitU0=False, fitU1=False, 
         U0_range=5, U1_range=5) :
  """ Creates NLEP_ parameters 

      Non Local External Potentials attempt to correct in part for the band-gap
      problem [*]_.

      :param type:
          Should be one of the following: 1|2|"liechtenstein"|"dudarev".
          Specifies the type of the Hubbard U. Defaults to 1.
      :param l:
          Should be one of the following: 0|1|2|3|"s"|"p"|"d"|"f".
          Channel for which to apply nlep. Defaults to 2.
      :param U0:
          First nlep parameter. Defaults to 0.
      :param U1:
          Second (e)nlep parameter. Defaults to 0.
      :param fitU0:
         If True, U1 is a free params w.r.t fitting potentials. 
         Only of use when performing NLEP fits.
      :param fitU1:
         If True, U1 is a free params w.r.t fitting potentials
         Only of use when performing NLEP fits.
      :param U0_range:
         Fitting bounds for U0.
         Only of use when performing NLEP fits.
      :param U1_range:
         Fitting bounds for U1.
         Only of use when performing NLEP fits.

      :note: LDA+U is always LSDA+U here. 

      .. _[*]: `PRB **77**, 241201(R) (2008). <NLEP_>`
      .. _NLEP: http://dx.doi.org/10.1103/PhysRevB.77.241201
  """
  ###from .. import vasp_has_nlep
  from ..error import ConfigError
  ###if not vasp_has_nlep: 
  ###  raise ConfigError(
  ###    'vasp_has_nlep is False. Cannot use NLEP.\n'
  ###    'If you have VASP compiled for NLEP, '
  ###    'please vasp_has_nlep to True in your '
  ###    'pylada configuration file.\n' )
  if hasattr(type, "lower"):
    type = type.lower()
    if type == "liechtenstein": type = 1
    elif type == "dudarev": type = 2
  if hasattr(l, "lower"):
    l = l.lower()
    assert len(l) == 1, "Uknown input %s." % (l)
    if   l[0] == 's': l = 0
    elif l[0] == 'p': l = 1
    elif l[0] == 'd': l = 2
    elif l[0] == 'f': l = 3
  try: l = int(l)
  except: raise ValueError("Moment l should be 0|1|2|3|s|p|d|f.")
  if l < 0 or l > 3: raise ValueError("Moment l should be 0|1|2|3|s|p|d|f.")
  if type != 1 and type != 2: raise ValueError("Unknown LDA+U type: {0}.".format(type))
  elif U1 is None: 
    return { "type": int(type), "l": l, "U0": U0, "func": "nlep",\
             "fitU0":fitU0,  "fitU":fitU0, "U_range":U0_range}
  else: 
    return { "type": int(type), "l": l, "U0": U0, "U1": U1, "func": "enlep",\
             "fitU0":fitU0, "fitU1":fitU1, "U0_range":U0_range, "U1_range":U1_range}

class Specie(object):
  """ Holds atomic specie information.
  
      Instances of this object define an atomic specie for VASP calculations.
      In addition, it may contain element-related information used to build a
      set of high-throughput jobs.
  """
  def __init__(self, directory, U=None, oxidation=None, **kwargs):
    """ Initializes a specie.

        :param directory:
            Directory with the potcar for this particular atomic types.  This
            directory should contain an *unzipped* POTCAR file.
        :param U:
            LDA+U parameters. It should a list of dictionaries, one entry per
            momentum channel, and each entry returned by a call to :py:func:`U`
            or :py:func:`nlep`.
        :param oxidation:
            Maximum oxidation state (or minimum if negative).
        :param kwargs:
            Any other keyworkd argument is added as an attribute of this object.
    """
    from ..misc import RelativePath

    self._directory = RelativePath(directory)
    if oxidation is not None: self.oxidation = oxidation
    if U is None: self.U = []
    elif isinstance(U, dict): self.U = [U]
    else: self.U = [u for u in U] # takes care of any kind of iterator.

    # sets up other arguments.
    for k, v in kwargs.items(): setattr(self, k, v)

  @property
  def directory(self):
    """ Directory where the POTCAR file may be found. """
    return self._directory.path
  @directory.setter
  def directory(self, value): self._directory.path = value

  @property 
  def path(self):
    """ Path to POTCAR file. """
    from os.path import join
    return join(self.directory, "POTCAR")

  @property
  def enmax(self):
    """ Maximum recommended cutoff """
    from quantities import eV
    import re
    self.potcar_exists()
    with self.read_potcar() as potcar:
      r = re.compile("ENMAX\s+=\s+(\S+);\s+ENMIN")
      p = r.search(potcar.read())
      if p is None: raise AssertionError, "Could not retrieve ENMAX from " + self.directory
      return float( p.group(1) ) * eV
  
  @property
  def valence(self):
    """ Number of valence electrons specified by pseudo-potential """ 
    self.potcar_exists()
    with self.read_potcar() as potcar:
      potcar.readline()
      return float(potcar.readline().split()[0]) # shoud be number on second line

  def potcar_exists(self):
    """ Raises IOError if POTCAR file does not exist. """
    from os.path import exists
    assert exists(self.path), IOError("Could not find POTCAR in {0}.".format(self.directory))

  def read_potcar(self):
    """ Returns handle/context to POTCAR file. """
    self.potcar_exists()
    return open(self.path, "r") 

  def __repr__(self):
    """ Represents a specie. """
    string = "{0.__class__.__name__}('{0._directory.unexpanded}'".format(self)
    for k, v in self.__dict__.items():
      if k[0] == '_directory': continue
      if k == 'U' and len(v) == 0: continue
      try: assert repr(v)[0] != '<' 
      except: continue
      string += ", {name}={value}".format(name=k, value=repr(v))
    return string + ')'
  
  def __setstate__(self, dictionary):
    """ Retrieves state from pickle.

        Takes care of older pickle as well.
    """
    self.__dict__.update(dictionary)
    if "path" in self.__dict__ and "_directory" not in self.__dict__:
      from ..misc import RelativePath
      self._directory = RelativePath(self.__dict__.pop("path"))
