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

""" Subpackage defining vasp incar parameters. """
__docformat__ = "restructuredtext en"
__all__ = [ "SpecialVaspParam", "ExtraElectron", "Algo", "Precision", "Ediff",\
            "Encut", "FFTGrid", "Restart", "UParams", "IniWave", 'Ediffg', "EncutGW", \
            "Incar", "Magmom", 'Npar', 'Boolean', 'Integer', 'Choices', 'PrecFock',
            "System", 'PartialRestart', 'Relaxation', 'Smearing', 'Lsorbit' ]
from _params import SpecialVaspParam, ExtraElectron, Algo, Precision, Ediff,\
                    Encut, FFTGrid, PartialRestart, Restart, UParams, IniWave, Magmom,\
                    Npar, Boolean, Integer, PrecFock, NonScf, Ediffg, Choices, \
                    EncutGW, System, Relaxation, Smearing, Lsorbit
from ...misc import add_setter

class Incar(object):
  """ Base class containing vasp input parameters.

      The following assumes you know how to write an INCAR. Although you won't
      need to anymore.  This class separates vasp parameters from methods to
      launch and control vasp.

      There are two kinds of parameters: 

        - Normal parameters which will simply print "NAME = VALUE" to the incar
        - Special parameters which enhance the default behavior of vasp
      
      The special parameters achieve a variety of design-goals. For instance,
      when passed a previous VASP run, :py:attr:`restart` will set ISTART_ and
      ICHARG_ accordingly, as well as copy the relevant files. If it is a
      :py:class:`Restart` object, it will also copy the CONTCAR file from the
      previous run (default behavior). If it as :py:class:`PartialRestart`,
      then the CONTCAR is not copied, which allows restarting from the charge
      with a slightly different structure than it was generated for.
  """
  def __init__(self): 
    super(Incar, self).__init__()

    # first, actually sets these two variables by hand, since they are used in __setattr__.
    super(Incar, self).__setattr__("params", {})
    super(Incar, self).__setattr__("special", {})
    self.add_param = "addgrid",     None
    self.add_param = "ispin",       1 
    self.add_param = "istart",      None
    self.add_param = "isym",        None
    self.add_param = "lmaxfockae",  None
    self.add_param = "lmaxmix",     4
    self.add_param = "lvhar",       False
    self.add_param = "lorbit",      None
    self.add_param = "nbands",      None
    self.add_param = "nomega",      None
    self.add_param = "nupdown",     None
    self.add_param = "symprec",     None
    # objects derived from SpecialVaspParams will be recognized as such and can
    # be added without further fuss.
    self.extraelectron = ExtraElectron(0)
    self.algo          = Algo()
    self.precision     = Precision("accurate")
    self.ediff         = Ediff(1e-4)
    self.ediffg        = Ediffg(None)
    self.encut         = Encut(None)
    self.encutgw       = EncutGW(None)
    self.fftgrid       = FFTGrid(None)
    self.restart       = Restart(None)
    self.U_verbosity   = UParams("occupancy")
    self.magmom        = Magmom()
    self.npar          = Npar(None)
    self.precfock      = PrecFock(None)
    self.nonscf        = NonScf(False)
    self.system        = System(True)
    self.smearing      = Smearing(None)
    self.relaxation    = Relaxation(None)
    self.lsorbit       = Lsorbit(None)

    self.lwave       = Boolean("lwave", False)
    self.lcharg      = Boolean("lcharg", True)
    self.lvtot       = Boolean("lvtot", False)
    self.lrpa        = Boolean("lrpa", None)
    self.loptics     = Boolean("loptics", None)
    self.lpead       = Boolean("lpead", None)
    self.lplane      = Boolean("lplane", None)
    self.nelm        = Integer("nelm", None)
    self.nelmin      = Integer("nelmin", None)
    self.nelmdl      = Integer("nelmdl", None)

  def incar_lines(self, **kwargs):
    """ List of incar lines. """

    # gathers special parameters.
    # Calls them first in case they change normal key/value pairs.
    result, specials, comments = [], [], []
    for key, value in self.special.items():
      if value.value is None: continue
      line = value.incar_string(**kwargs)
    # Then calls a second time in case they change each other.
    for key, value in self.special.items():
      if value.value is None: continue
      line = value.incar_string(**kwargs)
      if line is None: continue
      line = line.rstrip().lstrip()
      if line[-1] != '\n': line += '\n'
      if line[0] == '#': comments.append(line); continue
      if '=' in line and line.find('=') < 18:
        line = "{0: <{1}}".format(' ', 19 - line.find('=')) + line
      specials.append(line)
    # prints key/value pairs
    for key, value in self.params.items():
      if value is None: continue
      if isinstance(value, bool):  value = ".TRUE." if value else ".FALSE."
      else: 
        try: value = str(value)
        except ValueError: 
          raise ValueError("Could not convert vasp parameter {0} to string: {1}.".format(key, value))
      result.append( "{0: >18s} = {1}\n".format(key.upper(), value))
    # adds special parameter lines.
    result.extend(specials)
    result = sorted(result, key=lambda a: a.lstrip()[0])
    result.extend(comments)
    return result

  @add_setter
  def add_param(self, args):
    """ Adds/sets a vasp parameter.
    
        Consists of a key value pair. 

        >>> vasp.add_param = "ispin", 2

        This will result in the INCAR as "ISPIN = 2". Once set, the value can be accessed directly:

        >>> vasp.add_param = "ispin", 2
        >>> vasp.ispin = 1
        >>> print vasp.ispin # prints 1
    """
    key, value = args
    if isinstance(value, SpecialVaspParam):
      if key in self.params: del self.params[key] # one or other dictionary.
      self.special[key] = value
    else:
      if key in self.special: del self.special[key] # one or other dictionary.
      self.params[key] = value

  def __getattr__(self, name): 
    """ Gets a VASP parameter from standard and special dictionaries. """
    if name in self.params: return self.params[name]
    elif name in self.special: return self.special[name].value
    raise AttributeError("Unknown parameter " + name)

  def __setattr__(self, name, value):
    """ Sets a VASP parameter to standard and special dictionaries. """
    if isinstance(value, SpecialVaspParam):
      if name in self.params: del self.params[name]
      self.special[name] = value
    elif name in self.params: self.params[name] = value
    elif name in self.special: self.special[name].value = value
    else: super(Incar, self).__setattr__(name, value)

  def __delattr__(self, name): 
    """ Deletes a VASP parameter from standard and special dictionaries. """
    if name in self.__dict__: return self.__dict__.pop(name)
    elif name in self.params: return self.params.pop(name)
    elif name in self.params: return self.special.pop(name).value
    raise AttributeError("Unknown vasp attribute " + name + ".")

  def __dir__(self):
    result = [u for u in self.__dict__ if u[0] != '_'] 
    result.extend([u for u in self.params.keys() if u[0] != '_'])
    result.extend([u for u in self.special.keys() if u[0] != '_'])
    return list(set(result))

  @property
  def symmetries(self):
    """ Type of symmetry used in the calculation.
  
        This sets :py:attr:`isym` and :py:attr:`symprec` vasp tags. Can be
        "off" or a float corresponding to the tolerance used to determine
        symmetry operation. 
    """
    if self.isym is None and self.symprec is None: return True
    if self.isym is None: return self.symprec
    if self.isym == 0: return False

  @symmetries.setter
  def symmetries(self, value):
    if value is None: self.isym = None
    elif str(value).lower() == "off" or value is "0" or value is False: self.params["isym"] = 0
    elif str(value).lower() == "on" or value is True or value is True:
       self.symprec = None
       self.isym = None
    elif isinstance(value, float): 
       self.symprec = value
       self.isym = None
    else: raise ValueError("Uknown value when setting symmetries ({0}).".format(value))


  def __getstate__(self):
    d = self.__dict__.copy()
    params = d.pop("params")
    special = d.pop("special")
    return d, params, special
  def __setstate__(self, args):
    super(Incar, self).__setattr__("params", args[1])
    super(Incar, self).__setattr__("special", args[2])
    d = self.__dict__.update(args[0])
