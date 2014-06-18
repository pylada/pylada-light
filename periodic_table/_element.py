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

""" Defines Element class. """
__docformat__ = "restructuredtext en"

class Element(object):
  """ Contains atomic data for single element. 
  
      Data is taken from the `webelements`_ website.

      .. _webelements: http://www.webelements.com
  """
  def __init__(self, **kwargs):
    """ Initializes atoms """
    self.symbol                 = kwargs.pop('symbol', None)
    """ Atomic symbol. """
    self.name                   = kwargs.pop('name', None)
    """ Name of the element. """
    self.atomic_weight          = kwargs.pop('atomic_weight', None)
    """ Atomic weight (dimensionless) of the element. """
    self.atomic_number          = kwargs.pop('atomic_number', None)
    """ Atomic number (dimensionless) of the element. """
    self.pauling                = kwargs.pop('pauling', None)
    """ Pauling electronegativity (Pauling scale) of the element. """
    self.sanderson              = kwargs.pop('sanderson', None)
    """ Sanderson electronegativity (Pauling scale) of the element. """
    self.allred_rochow          = kwargs.pop('allred_rochow', None)
    """ Allred-Rochow electronegativity (Pauling scale) of the element. """
    self.mulliken_jaffe         = kwargs.pop('mulliken_jaffe', None)
    """ Mulliken-Jaffe electronegativity (Pauling scale) of the element. """
    self.allen                  = kwargs.pop('allen', None)
    """ Allen electronegativity (Pauling scale) of the element. """
    self.electron_affinity      = kwargs.pop('electron_affinity', None)
    """ Electron affinity of the element (kJ per mol). """
    self.ionization_energies    = kwargs.pop('ionization_energies', None)
    """ Known Ionization energies of the element (kJ per mol).
    
        All ionization energies known to www.webelements.com are listed, from
        first to last.
    """
    self.atomic_radius          = kwargs.pop('atomic_radius', None)
    """ Empirical atomic radius. """
    self.covalent_radius        = kwargs.pop('covalent_radius', None)
    """ Covalent bond radius. """
    self.single_bond_radius     = kwargs.pop('single_bond_radius', None)
    """ Single covalent-bond  radius. """
    self.double_bond_radius     = kwargs.pop('double_bond_radius', None)
    """ Double covalent-bond  radius. """
    self.triple_bond_radius     = kwargs.pop('triple_bond_radius', None)
    """ Triple covalent-bond  radius. """
    self.van_der_waals_radius   = kwargs.pop('van_der_waals_radius', None)
    """ van der Walls radius. """
    self.fusion                 = kwargs.pop('fusion', None)
    """ Enthalpy of fusion. """
    self.vaporization           = kwargs.pop('vaporization', None)
    """ Enthalpy of vaporization. """
    self.atomization            = kwargs.pop('atomization', None)
    """ Enthalpy of atomization. """
    self.melting_point          = kwargs.pop('melting_point', None)
    """ Melting point of the elemental solid. """
    self.boiling_point          = kwargs.pop('boiling_point', None)
    """ Boiling point of the elemental solid. """
    self.critical_temperature   = kwargs.pop('critical_temperature', None)
    """ Critical temperature of the elemental solid. """
    self.thermal_conductivity   = kwargs.pop('thermal_conductivity', None)
    """ Thermal conductivity of the elemental solid. """
    self.thermal_expansion      = kwargs.pop('thermal_expansion', None)
    """ Coefficient of lineary thermal expansion of the elemental solid at 273K. """
    self.density                = kwargs.pop('density', None)
    """ Density of the elemental solid. """
    self.molar_volume           = kwargs.pop('molar_volume', None)
    """ Molar volume of the element at 298K. """
    self.sound_velocity         = kwargs.pop('sound_velocity', None)
    """ Velocity of sound in the element at 298K. """
    self.young_modulus          = kwargs.pop('young_modulus', None)
    """ Young modulus of the elemental solid. """
    self.rigidity_modulus       = kwargs.pop('rigidity_modulus', None)
    """ Rigidity modulus of the elemental solid. """
    self.bulk_modulus           = kwargs.pop('bulk_modulus', None)
    """ Bulk modulus ratio of the elemental solid. """
    self.poisson_ratio          = kwargs.pop('poisson_ratio', None)
    """ Poisson ratio of the elemental solid. """
    self.electrical_resistivity = kwargs.pop('electical_resistivity', None)
    """ Electrical Resistivity ratio of the elemental solid. """
    self.pettifor               = kwargs.pop('pettifor', None)
    """ This element on the Pettifor_ scale.

        Pettifor_'s is an artificial scale designed to parameterizes
        a two-dimensional structure map of binary AB compounds. 
  
        References
        ==========
          .. _Pettifor : D.G. Pettifor, Solid. Stat. Comm., *51* 31-34 (1984).
    """
    self.orbital_radii        = kwargs.pop('orbital_radii', None)
    """ Orbital_ radii of this element.

        The orbital radii can be defined for s, p, and d orbitals using
        psudo-potential wavefunctions. 
        
        References
        ==========
       
        .. _Orbital : Alex Zunger, PRB *22* 5839-5872 (1980),
            http://dx.doi.org/10.1103/PhysRevB.22.5839
    """



  def __str__(self):
    string = "{0} elemental data.\n\n".format(self.name)
    for name in self.__dict__:
      if name[0] == '_': continue
      s = getattr(self, name)
      if s is not None: string += "  - {0}: {1}.\n".format(name, str(s))
    
    return string

  @property
  def electronic_configuration(self):
    """ Returns array describing electronic configuration.

        The first element of the array discribe 1s electrons, the second 2s and
        2p, and so forth.
        Each element is a dictionary where each key is an orbital and each
        value the corresponding number of electrons in that orbital.
    """
    N = self.atomic_number
    if N < 2: return [{'s': 1}]
    result = [{'s': 2}]
    if N < 3: return result
    result.append({'s': min(N-2, 2), 'p': min(N-4, 6) if N > 4 else 0})
    if N < 11: return result
    result.append({'s': min(N-10, 2), 'p': min(N-12, 6) if N > 12 else 0})
    if N < 19: return result
    if N == 21:
      result.append({'s': 2, 'p': 0, 'd':1})
      return result
    if N == 24:
      result.append({'s': 1, 'p': 0, 'd':5})
      return result
    if N == 29:
      result.append({'s': 1, 'p': 0, 'd':10})
      return result
    result.append({'s': min(N-18, 2), 'p': min(N-30, 6) if N > 30 else 0, 'd': min(N-20, 10) if N > 20 else 0})
    if N < 37: return result
    if N == 39:
      result.append({'s': 2, 'p': 0, 'd':1})
      return result
    if N == 41:
      result.append({'s': 1, 'p': 0, 'd':4})
      return result
    if N == 42:
      result.append({'s': 1, 'p': 0, 'd':5})
      return result
    if N == 44:
      result.append({'s': 1, 'p': 0, 'd':7})
      return result
    if N == 45:
      result.append({'s': 1, 'p': 0, 'd':8})
      return result
    if N == 47:
      result.append({'s': 1, 'p': 0, 'd':10})
      return result
    result.append({'s': min(N-36, 2), 'p': min(N-48, 6) if N > 48 else 0, 'd': min(N-38, 10) if N > 38 else 0})
    if N < 55: return result
    if N == 57:
      result.append({'s': 2, 'p': 0, 'd': 1, 'f': 0})
      return result
    if N == 58:
      result.append({'s': 2, 'p': 0, 'd': 1, 'f': 1})
      return result
    if N == 64:
      result.append({'s': 2, 'p': 0, 'd': 1, 'f': 7})
      return result
    if N == 71:
      result.append({'s': 2, 'p': 0, 'd': 1, 'f': 14})
      return result
    if N == 78:
      result.append({'s': 1, 'p': 0, 'd':9, 'f': 14})
      return result
    if N == 79:
      result.append({'s': 1, 'p': 0, 'd':10, 'f': 14})
      return result
    result.append({'s': min(N-54, 2), 'p': min(N-80, 6) if N > 80 else 0,
                   'd': min(N-70, 10) if N > 70 else 0, 
                   'f': min(N-56, 14) if N > 56 else 0})
    if N < 87: return result
    if N == 89:
      result.append({'s': 2, 'p': 0, 'd':1, 'f': 0})
      return result
    if N == 91:
      result.append({'s': 2, 'p': 0, 'd':1, 'f': 2})
      return result
    if N == 92:
      result.append({'s': 2, 'p': 0, 'd':1, 'f': 3})
      return result
    if N == 93:
      result.append({'s': 2, 'p': 0, 'd':1, 'f': 4})
      return result
    if N == 96:
      result.append({'s': 2, 'p': 0, 'd':1, 'f': 7})
      return result
    if N == 103:
      result.append({'s': 2, 'p': 0, 'd':1, 'f': 14})
      return result
    result.append({'s': min(N-86, 2), 'p': 0, 
                   'd': min(N-102, 10) if N > 102 else 0, 
                   'f': min(N-88, 14) if N > 88 else 0})
    return result

  @property 
  def group(self):
    """ Group, eg column in periodict table. """
    last = self.electronic_configuration[-1]
    s = last.get('s', 0)
    p = last.get('p', 0)
    d = last.get('d', 0)
    if self.atomic_number > 57 and self.atomic_number < 72: return "Lanthanide"
    elif self.atomic_number > 89 and self.atomic_number < 104: return "Actinide"
    elif s == 2 and p == 6: return "VIII"
    elif s+p+d == 1: return "IA"
    elif s+p+d == 2: return "IIA"
    elif s+p ==  3: return "IIIB"
    elif s+d ==  3: return "IIIA"
    elif s+p ==  4: return "IVB"
    elif s+d ==  4: return "IVA"
    elif s+p ==  5: return "VB"
    elif s+d ==  5: return "VA"
    elif s+p ==  6: return "VIB"
    elif s+d ==  6: return "VIA"
    elif s+p ==  7: return "VIIB"
    elif s+d ==  7: return "VIIA"
    elif s+d in [8, 9, 10]: return "VIIIA"
    elif s+d == 11: return "IA"
    elif s+d == 12: return "IIA"

  @property
  def column(self):
    """ Returns column in periodic table. """
    if self.atomic_number > 57 and self.atomic_number < 72: return "Lanthanide"
    elif self.atomic_number > 89 and self.atomic_number < 104: return "Actinide"
    last = self.electronic_configuration[-1]
    return last.get('s', 0) + last.get('p', 0) + last.get('d', 10)

  @property
  def row(self):
    """ Returns row in periodic table. """
    return len(self.electronic_configuration)

  def __repr__(self):
    result = {}
    for name in self.__dict__:
      if name[0] == '_': continue
      s = getattr(self, name)
      if s is not None: result[name] = s
    return "{0}(**{1})".format(self.__class__.__name__, result)
