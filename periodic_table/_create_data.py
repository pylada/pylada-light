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

""" Creates data by trolling webelements.com. """
__docformat__ = "restructuredtext en"
import quantities as qt
from ._element import Element

def _orbital_radii():
  """ Zunger `Orbital radii`_.
  

      The orbital radii (in a.u.) are defined for s, p, and d orbitals using
      psudo-potential wavefunctions.
      
      References
      ==========

      .. _Orbital radii : Alex Zunger, PRB *22* 5839-5872 (1980),
          http://dx.doi.org/10.1103/PhysRevB.22.5839

  """
  return  { "Li":( 0.985, 0.625 ), 
            "Be":(  0.64,  0.44 ), 
             "B":(  0.48, 0.315 ), 
             "C":(  0.39,  0.25 ), 
             "N":(  0.33,  0.21 ), 
             "O":( 0.285,  0.18 ), 
             "F":(  0.25, 0.155 ), 
            "Ne":(  0.22,  0.14 ), 
            
            "Na":(  1.10,  1.55 ), 
            "Mg":(  0.90,  1.13 ), 
            "Al":(  0.77, 0.905 ), 
            "Si":(  0.68,  0.74 ),  
             "P":(  0.60,  0.64 ), 
             "S":(  0.54,  0.56 ), 
            "Cl":(  0.50,  0.51 ), 
            "Ar":(  0.46,  0.46 ), 
                   
             "K":(  1.54,  2.15,  0.37 ),
            "Ca":(  1.32,  1.68,  0.34 ),
            "Sc":(  1.22,  1.53,  0.31 ),
            "Ti":(  1.15,  1.43,  0.28 ),
             "V":(  1.09,  1.34,  0.26 ),
            "Cr":(  1.07,  1.37,  0.25 ),
            "Mn":(  0.99,  1.23,  0.23 ),
            "Fe":(  0.95,  1.16,  0.22 ),
            "Co":(  0.92,  1.10,  0.21 ),
            "Ni":(  0.96,  1.22, 0.195 ),
            "Cu":(  0.88,  1.16, 0.185 ),
            "Zn":(  0.82,  1.06, 0.175 ),
            "Ga":(  0.76, 0.935,  0.17 ),
            "Ge":(  0.72,  0.84,  0.16 ),
            "As":(  0.67, 0.745, 0.155 ),
            "Se":( 0.615,  0.67,  0.15 ),
            "Br":(  0.58,  0.62, 0.143 ),
            "Kr":(  0.56,  0.60, 0.138 ), 

            "Rb":(  1.67,  2.43,  0.71 ),
            "Sr":(  1.42,  1.79, 0.633 ),
             "Y":(  1.32,  1.62,  0.58 ),
            "Zr":( 1.265,  1.56,  0.54 ),
            "Nb":(  1.23,  1.53,  0.51 ),
            "Mo":(  1.22,  1.50,  0.49 ),
            "Tc":(  1.16,  1.49, 0.455 ),
            "Ru":( 1.145,  1.46,  0.45 ),
            "Rh":(  1.11,  1.41,  0.42 ),
            "Pd":(  1.08,  1.37,  0.40 ), 
            "Ag":( 1.045,  1.33, 0.385 ),
            "Cd":( 0.985,  1.23,  0.37 ),
            "In":(  0.94,  1.11,  0.36 ),
            "Sn":(  0.88,  1.00, 0.345 ),
            "Sb":(  0.83, 0.935, 0.335 ),
            "Te":(  0.79,  0.88, 0.325 ),
             "I":( 0.755,  0.83, 0.315 ),
            "Xe":(  0.75,  0.81, 0.305 ),
          
            "Cs":(  1.71,  2.60 ),
            "Ba":( 1.515, 1.887,  0.94 ),
            "La":( 1.375, 1.705, 0.874 ),
            "Hf":(  1.30,  1.61,  0.63 ),
            "Ta":(  1.25,  1.54, 0.605 ),
             "W":(  1.22, 1.515,  0.59 ), 
            "Re":(  1.19,  1.49, 0.565 ),
            "Os":(  1.17,  1.48, 0.543 ),
            "Ir":(  1.16, 1.468, 0.526 ),
            "Pt":(  1.24,  1.46,  0.51 ),
            "Au":(  1.21,  1.45, 0.488 ),
            "Hg":(  1.07,  1.34, 0.475 ),
            "Tl":( 1.015,  1.22, 0.463 ),
            "Pb":(  0.96,  1.13,  0.45 ),
            "Bi":(  0.92, 1.077, 0.438 ),
            "Po":(  0.88,  1.02, 0.425 ),
            "At":(  0.85,  0.98, 0.475 ),
            "Rn":(  0.84,  0.94, 0.405 )  }

def _pettifor_numbers():
  """ Pettifor numbers. 
  
      The `Pettifor numbers`_ make up scale which parameterizes a two-dimensional
      structure map of binary AB compounds. 

      References
      ==========
        .. _Pettifor numbers : D.G. Pettifor, Solid. Stat. Comm., *51* 31-34 (1984).
  """
  return  { "Li": 0.45,
            "Be": 1.5,
             "B": 2.0,
             "C": 2.5,
             "N": 3.0, 
             "O": 3.5,
             "F": 4.0,
            
            "Na": 0.4,
            "Mg": 1.28,
            "Al": 1.66,
            "Si": 1.92,
             "P": 2.18,
             "S": 2.44,
            "Cl": 2.70,
                   
             "K": 0.35,
            "Ca": 0.60,
            "Sc": 0.74,
            "Ti": 0.79,
             "V": 0.84,
            "Cr": 0.89,
            "Mn": 0.94,
            "Fe": 0.99,
            "Co": 1.04,
            "Ni": 1.09,
            "Cu": 1.20,
            "Zn": 1.44,
            "Ga": 1.68,
            "Ge": 1.92,
            "As": 2.16,
            "Se": 2.40,
            "Br": 2.64,

            "Rb": 0.30,
            "Sr": 0.55,
             "Y": 0.70,
            "Zr": 0.76,
            "Nb": 0.82,
            "Mo": 0.88,
            "Tc": 0.94,
            "Ru": 1.00,
            "Rh": 1.06,
            "Pd": 1.12,
            "Ag": 1.18,
            "Cd": 1.36,
            "In": 1.60,
            "Sn": 1.84,
            "Sb": 2.08,
            "Te": 2.32,
             "I": 2.56,
          
            "Cs": 0.25,
            "Ba": 0.50,
            "La": 0.748,
            "Hf": 0.775,
            "Ta": 0.83,
             "W": 0.885,
            "Re": 0.94,
            "Os": 0.995,
            "Ir": 1.05,
            "Pt": 1.105,
            "Au": 1.16,
            "Hg": 1.32,
            "Tl": 1.56,
            "Pb": 1.80,
            "Bi": 2.04,
            "Po": 2.28, 
            "At": 2.52 }

def _download_files():
  """ Downloads data from webelements.com. """
  import urllib
  from os import makedirs
  from os.path import exists, join
  
  atom_list = ['Ruthenium', 'Rhenium', 'Rutherfordium', 'Radium', 'Rubidium',
    'Radon', 'Rhodium', 'Beryllium', 'Barium', 'Bohrium', 'Bismuth',
    'Berkelium', 'Bromine', 'Hydrogen', 'Phosphorus', 'Osmium', 'Mercury',
    'Germanium', 'Gadolinium', 'Gallium', 'Ununbium', 'Praseodymium',
    'Platinum', 'Plutonium', 'Carbon', 'Lead', 'Protactinium', 'Palladium',
    'Xenon', 'Polonium', 'Promethium', 'Hassium',
    'Holmium', 'Hafnium', 'Molybdenum', 'Helium', 'Mendelevium', 'Magnesium',
    'Potassium', 'Manganese', 'Oxygen', 'Meitnerium', 'Sulfur', 'Tungsten',
    'Zinc', 'Europium', 'Einsteinium', 'Erbium', 'Nickel', 'Nobelium',
    'Sodium', 'Niobium', 'Neodymium', 'Neon', 'Neptunium', 'Francium', 'Iron',
    'Fermium', 'Boron', 'Fluorine', 'Strontium', 'Nitrogen', 'Krypton',
    'Silicon', 'Tin', 'Samarium', 'Vanadium', 'Scandium', 'Antimony',
    'Seaborgium', 'Selenium', 'Cobalt', 'Curium', 'Chlorine', 'Calcium',
    'Californium', 'Cerium', 'Cadmium', 'Thulium', 'Caesium', 'Chromium',
    'Copper', 'Lanthanum', 'Lithium', 'Thallium', 'Lutetium', 'Lawrencium',
    'Thorium', 'Titanium', 'Tellurium', 'Terbium', 'Technetium', 'Tantalum',
    'Ytterbium', 'Dubnium', 'Zirconium', 'Dysprosium', 'Iodine', 'Uranium',
    'Yttrium', 'Actinium', 'Silver', 'Iridium', 'Americium', 'Aluminium',
    'Arsenic', 'Argon', 'Gold', 'Astatine', 'Indium', 'Darmstadtium', 'Copernicium']

  if not exists("elements"): makedirs("elements")
  for name in atom_list: 
    file = urllib.urlopen("http://www.webelements.com/{0}".format(name.lower()))
    string = file.read()
    file.close()
    with open(join("elements", name), "w") as out: out.write(string)
    file = urllib.urlopen("http://www.webelements.com/{0}/atoms.html".format(name.lower()))
    string = file.read()
    file.close()
    with open(join("elements", name + "_atoms.html"), "w") as out: out.write(string)
    file = urllib.urlopen( "http://www.webelements.com/{0}/electronegativity.html"\
                           .format(name.lower()))
    string = file.read()
    file.close()
    with open(join("elements", name + "_electronegativity.html"), "w") as out: out.write(string)
    file = urllib.urlopen( "http://www.webelements.com/{0}/atom_sizes.html"\
                           .format(name.lower()))
    string = file.read()
    file.close()
    with open(join("elements", name + "_atom_sizes.html"), "w") as out: out.write(string)
    file = urllib.urlopen( "http://www.webelements.com/{0}/thermochemistry.html"\
                           .format(name.lower()))
    string = file.read()
    file.close()
    with open(join("elements", name + "_thermochemistry.html"), "w") as out: out.write(string)
    file = urllib.urlopen( "http://www.webelements.com/{0}/physics.html"\
                           .format(name.lower()))
    string = file.read()
    file.close()
    with open(join("elements", name + "_physics.html"), "w") as out: out.write(string)


def _create_elements_py(filename="_elements.py"):
  """ Gets data from webelements.com and creates _elements.py. """
  import re
  from pickle import dumps
  import urllib
  from os.path import exists, join
  from BeautifulSoup import BeautifulSoup, HTMLParseError
  from ..physics import a0
  import quantities as pq

  atom_list = [ # 'Silicon', 'Hydrogen', 'Gold' ] 
               'Ruthenium', 'Rhenium', 'Rutherfordium', 'Radium', 'Rubidium',
               'Radon', 'Rhodium', 'Beryllium', 'Barium', 'Bohrium', 'Bismuth',
               'Berkelium', 'Bromine', 'Hydrogen', 'Phosphorus', 'Osmium', 'Mercury',
               'Germanium', 'Gadolinium', 'Gallium', 'Ununbium', 'Praseodymium',
               'Platinum', 'Plutonium', 'Carbon', 'Lead', 'Protactinium', 'Palladium',
               'Xenon', 'Polonium', 'Promethium', 'Hassium',
               'Holmium', 'Hafnium', 'Molybdenum', 'Helium', 'Mendelevium', 'Magnesium',
               'Potassium', 'Manganese', 'Oxygen', 'Meitnerium', 'Sulfur', 'Tungsten',
               'Zinc', 'Europium', 'Einsteinium', 'Erbium', 'Nickel', 'Nobelium',
               'Sodium', 'Niobium', 'Neodymium', 'Neon', 'Neptunium', 'Francium', 'Iron',
               'Fermium', 'Boron', 'Fluorine', 'Strontium', 'Nitrogen', 'Krypton',
               'Silicon', 'Tin', 'Samarium', 'Vanadium', 'Scandium', 'Antimony',
               'Seaborgium', 'Selenium', 'Cobalt', 'Curium', 'Chlorine', 'Calcium',
               'Californium', 'Cerium', 'Cadmium', 'Thulium', 'Caesium', 'Chromium',
               'Copper', 'Lanthanum', 'Lithium', 'Thallium', 'Lutetium', 'Lawrencium',
               'Thorium', 'Titanium', 'Tellurium', 'Terbium', 'Technetium', 'Tantalum',
               'Ytterbium', 'Dubnium', 'Zirconium', 'Dysprosium', 'Iodine', 'Uranium',
               'Yttrium', 'Actinium', 'Silver', 'Iridium', 'Americium', 'Aluminium',
               'Arsenic', 'Argon', 'Gold', 'Astatine', 'Indium']

  orbital_radii = _orbital_radii()
  pettifor_numbers = _pettifor_numbers()

  re_swf = re.compile("(rainbow|NI3|volcano|\_flash|K\_H2O).swf\s*(?!\")")
  re_atomweight = re.compile(":\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?")
  results = {}
  for name in atom_list: 

    # first opens and reads file.
    if not exists(join("elements", name)): 
      file = urllib.urlopen("http://www.webelements.com/{0}".format(name.lower()))
      string = file.read()
      file.close()
    else:
      with open(join("elements", name), "r") as file: string = file.read()
    string = string.replace("alt\"", "alt=\"")
    soup = BeautifulSoup(re.sub(re_swf,"rainbow.swf\"",string))

    atom = Element(name=name)
    atom.symbol = soup.findChild( name="a", attrs={"title": "Element names and symbols"},\
                                  text=" Symbol").parent.parent.contents[1].split()[1]
    atom.atomic_number = soup.findChild(name="a", attrs={"title": "Element atomic numbers"})\
                                       .parent.contents[-1].split()[1]
    atom.atomic_number = int(atom.atomic_number)
    atom.atomic_weight = soup.findChild(name="a", attrs={"title": "Element atomic weights"})\
                                       .parent.prettify()
    found = re_atomweight.search(atom.atomic_weight)
    if found is None: print name
    else: atom.atomic_weight = float(found.group(1))

    
    # ionization stuff
    if not exists(join("elements", name + "_atoms.html")):
      file = urllib.urlopen("http://www.webelements.com/{0}/atoms.html".format(name.lower()))
      string = file.read()
      file.close()
    else: 
      with open(join("elements", name + "_atoms.html"), "r") as file: string = file.read()
    soup = BeautifulSoup(string) 
    # electron affinity
    found = re.search("of\s+{0}\s+is\s+(\S+)".format(name.lower()), string)
    if found.group(1) == "no": atom.electron_affinity = None
    else: atom.electron_affinity = float(found.group(1)) * pq.kilo * pq.J / pq.mol
    # ionization energies
    energies = []
    for child in soup.findChild(name="table", attrs={"class":"chemistry-data"})\
                     .findChildren(name='td'):
      energies.append(float(child.string) * pq.kilo * pq.J / pq.mol)
    atom.ionization_energies = energies if len(energies) > 0 else None


    # electronegativities.
    if not exists(join("elements", name + "_electronegativity.html")):
      file = urllib.urlopen("http://www.webelements.com/{0}/electronegativity.html"\
                            .format(name.lower()))
      string = file.read()
      file.close()
    else: 
      with open(join("elements", name + "_electronegativity.html"), "r") as file:
          string = file.read()
    soup = BeautifulSoup(string) 
    attrs = { "href": "../periodicity/electronegativity_pauling/",\
              "title": "View definition and pictures showing periodicity "\
                       "of Pauling electronegativity"}
    pauling = soup.findChild(name="a", attrs=attrs).parent.parent.contents[-1].string
    pauling = pauling.split()[0]
    atom.pauling = float(pauling) if pauling != "no" else None

    attrs = { "href": "../periodicity/electronegativity_sanderson/" }
    sanderson = soup.findChild(name="a", attrs=attrs).parent.parent.contents[-1].string
    sanderson = sanderson.split()[0]
    atom.sanderson = float(sanderson) if sanderson != "no" else None

    attrs = { "href": "../periodicity/electroneg_allred_rochow/" }
    allred_rochow = soup.findChild(name="a", attrs=attrs).parent.parent.contents[-1].string
    allred_rochow = allred_rochow.split()[0]
    atom.allred_rochow = float(allred_rochow) if allred_rochow != "no" else None

    attrs = { "href": "../periodicity/electroneg_mulliken_jaffe/" }
    mulliken_jaffe = soup.findChild(name="a", attrs=attrs).parent.parent.contents[-1]
    if name in ["Germanium", "Gallium", "Carbon", "Lead", "Boron", "Silicon", "Tin",\
                "Thallium", "Aluminium", "Indium"]: 
      mulliken_jaffe = mulliken_jaffe.contents[0]
    else: mulliken_jaffe = mulliken_jaffe.string
    mulliken_jaffe = mulliken_jaffe.split()[0]
    atom.mulliken_jaffe = float(mulliken_jaffe) if mulliken_jaffe != "no" else None

    attrs = { "href": "../periodicity/electronegativity_allen/" }
    allen = soup.findChild(name="a", attrs=attrs).parent.parent.contents[-1].string
    allen = allen.split()[0]
    atom.allen = float(allen) if allen != "no" else None
    
    # atom sizes
    if not exists(join("elements", name + "_atom_sizes.html")):
      file = urllib.urlopen("http://www.webelements.com/{0}/atom_sizes.html"\
                            .format(name.lower()))
      string = file.read()
      file.close()
    else: 
      with open(join("elements", name + "_atom_sizes.html"), "r") as file:
          string = file.read()
    soup = BeautifulSoup(string) 
    
    # atomic radius
    attrs = { "href": "../periodicity/atomic_radius_empirical/" }
    atomic_radius = soup.findChild(name="a", attrs=attrs).parent.contents[-1].split()[1]
    if atomic_radius != "no":
      atom.atomic_radius = float(atomic_radius) * pq.picometre 
    
    attrs = { "href": "../periodicity/covalent_radius_2008/" }
    covalent_radius = soup.findChild(name="a", attrs=attrs).parent.contents[-1].split()[1]
    atom.covalent_radius = float(covalent_radius) * pq.picometre if covalent_radius != "no" else None

    attrs = { "href": "../periodicity/radii_covalent_single/" }
    single_bond_radius = soup.findChild(name="a", attrs=attrs)
    if single_bond_radius is not None:
      single_bond_radius = single_bond_radius.parent.contents[-1].split()[1]
      if single_bond_radius != "no": 
        atom.single_bond_radius = float(single_bond_radius) * pq.picometre

    attrs = { "href": "../periodicity/radii_covalent_double/" }
    double_bond_radius = soup.findChild(name="a", attrs=attrs)
    if double_bond_radius is not None:
      double_bond_radius = double_bond_radius.parent.contents[-1].split()[1]
      if double_bond_radius != "no": 
        atom.double_bond_radius = float(double_bond_radius) * pq.picometre

    attrs = { "href": "../periodicity/radii_covalent_triple/" }
    triple_bond_radius = soup.findChild(name="a", attrs=attrs)
    if triple_bond_radius is not None:
      triple_bond_radius = triple_bond_radius.parent.contents[-1].split()[1]
      if triple_bond_radius != "no": 
        atom.triple_bond_radius = float(triple_bond_radius) * pq.picometre

    attrs = { "href": "../periodicity/van_der_waals_radius/" }
    van_der_waals_radius = soup.findChild(name="a", attrs=attrs)
    if van_der_waals_radius is not None:
      van_der_waals_radius = van_der_waals_radius.parent.contents[-1].split()[1]
      if van_der_waals_radius != "no": 
        atom.van_der_waals_radius = float(van_der_waals_radius) * pq.picometre

    # thermochemistry
    if not exists(join("elements", name + "_thermochemistry.html")):
      file = urllib.urlopen("http://www.webelements.com/{0}/thermochemistry.html"\
                            .format(name.lower()))
      string = file.read()
      file.close()
    else: 
      with open(join("elements", name + "_thermochemistry.html"), "r") as file:
          string = file.read()
    soup = BeautifulSoup(string) 
    
    attrs = { "href": "../periodicity/enthalpy_fusion/" }
    fusion = soup.findChild(name="a", attrs=attrs).parent.prettify()
    fusion = re.search(":\s*(?:about)?\s*(\S+)", fusion)
    if fusion is not None and fusion.group(1) != "no":
      atom.fusion = float(fusion.group(1)) * pq.kilo * pq.J / pq.mol 

    attrs = { "href": "../periodicity/enthalpy_vaporisation/" }
    vaporization = soup.findChild(name="a", attrs=attrs).parent.prettify()
    vaporization = re.search(":\s*(?:about)?\s*(\S+)", vaporization)
    if vaporization is not None and vaporization.group(1) != "no":
      atom.vaporization = float(vaporization.group(1)) * pq.kilo * pq.J / pq.mol 

    attrs = { "href": "../periodicity/enthalpy_atomisation/" }
    atomization = soup.findChild(name="a", attrs=attrs).parent.prettify()
    atomization = re.search(":\s*(?:about)?\s*(\S+)", atomization)
    if atomization is not None and atomization.group(1) != "no":
      atom.atomization = float(atomization.group(1)) * pq.kilo * pq.J / pq.mol 

    # physics
    if not exists(join("elements", name + "_physics.html")):
      file = urllib.urlopen("http://www.webelements.com/{0}/physics.html"\
                            .format(name.lower()))
      string = file.read()
      file.close()
    else: 
      with open(join("elements", name + "_physics.html"), "r") as file:
          string = file.read()
    soup = BeautifulSoup(string) 

    attrs = { "href": "../periodicity/melting_point/" }
    melting_point = soup.findChild(name="a", attrs=attrs).parent.prettify()
    melting_point = re.search(":\s*(?:\(white P\)|about|maybe about)?\s*(\S+)", melting_point)
    if melting_point is not None and melting_point.group(1) != "no":
      atom.melting_point = float(melting_point.group(1)) * pq.Kelvin

    attrs = { "href": "../periodicity/boiling_point/" }
    boiling_point = soup.findChild(name="a", attrs=attrs).parent.prettify()
    boiling_point = re.search(":\s*(?:about)?\s*(\S+)", boiling_point)
    if boiling_point is not None and boiling_point.group(1) != "no":
      atom.boiling_point = float(boiling_point.group(1)) * pq.Kelvin

    attrs = { "href": "../periodicity/critical_temperature/" }
    critical_temperature = soup.findChild(name="a", attrs=attrs).parent.prettify()
    critical_temperature = re.search(":\s*(?:about)?\s*(\S+)", critical_temperature)
    if critical_temperature is not None and critical_temperature.group(1) != "no":
      atom.critical_temperature = float(critical_temperature.group(1)) * pq.Kelvin

    attrs = { "href": "../periodicity/thermal_conductivity/" }
    thermal_conductivity = soup.findChild(name="a", attrs=attrs).parent.prettify()
    thermal_conductivity = re.search(":\s*(?:about)?\s*(\S+)", thermal_conductivity)
    if thermal_conductivity is not None and thermal_conductivity.group(1) != "no":
      atom.thermal_conductivity = float(thermal_conductivity.group(1)) * pq.W / pq.m / pq.K

    attrs = { "href": "../periodicity/coeff_thermal_expansion/" }
    thermal_expansion = soup.findChild(name="a", attrs=attrs).parent.prettify()
    thermal_expansion = re.search(":\s*(?:about)?\s*(\S+)", thermal_expansion)
    if thermal_expansion is not None and thermal_expansion.group(1) != "no":
      atom.thermal_expansion = float(thermal_expansion.group(1)) * pq.micro / pq.K

    attrs = { "href": "../periodicity/density/" }
    density = soup.findChild(name="a", attrs=attrs).parent.prettify()
    density = re.search(":\s*(?:about)?\s*(\S+)", density)
    if density is not None and density.group(1) != "no":
      atom.density = float(density.group(1)) / 1000 * pq.g * pq.cm**3

    attrs = { "href": "../periodicity/molar_volume/" }
    molar_volume = soup.findChild(name="a", attrs=attrs).parent.prettify()
    molar_volume = re.search(":\s*(?:about)?\s*(\S+)", molar_volume)
    if molar_volume is not None and molar_volume.group(1) != "no":
      atom.molar_volume = float(molar_volume.group(1)) * pq.cm**3 / pq.mol

    attrs = { "href": "../periodicity/velocity_sound/" }
    sound_velocity = soup.findChild(name="a", attrs=attrs).parent.prettify()
    sound_velocity = re.search(":\s*(?:about)?\s*(\S+)", sound_velocity)
    if sound_velocity is not None and sound_velocity.group(1) != "no":
      atom.sound_velocity = float(sound_velocity.group(1)) * pq.m / pq.s

    attrs = { "href": "../periodicity/youngs_modulus/" }
    young_modulus = soup.findChild(name="a", attrs=attrs).parent.prettify()
    young_modulus = re.search(":\s*(?:about)?\s*(\S+)", young_modulus)
    if young_modulus is not None and young_modulus.group(1) != "no":
      atom.young_modulus = float(young_modulus.group(1)) * pq.GPa

    attrs = { "href": "../periodicity/rigidity_modulus/" }
    rigidity_modulus = soup.findChild(name="a", attrs=attrs).parent.prettify()
    rigidity_modulus = re.search(":\s*(?:about)?\s*(\S+)", rigidity_modulus)
    if rigidity_modulus is not None and rigidity_modulus.group(1) != "no":
      atom.rigidity_modulus = float(rigidity_modulus.group(1)) * pq.GPa
    
    attrs = { "href": "../periodicity/bulk_modulus/" }
    bulk_modulus = soup.findChild(name="a", attrs=attrs).parent.prettify()
    bulk_modulus = re.search(":\s*(?:about)?\s*(\S+)", bulk_modulus)
    if bulk_modulus is not None and bulk_modulus.group(1) != "no":
      atom.bulk_modulus = float(bulk_modulus.group(1)) * pq.GPa
    
    attrs = { "href": "../periodicity/poissons_ratio/" }
    poisson_ratio = soup.findChild(name="a", attrs=attrs).parent.prettify()
    poisson_ratio = re.search(":\s*(?:about)?\s*(\S+)", poisson_ratio)
    if poisson_ratio is not None and poisson_ratio.group(1) != "no":
      atom.poisson_ratio = float(poisson_ratio.group(1)) * pq.dimensionless
    
    attrs = { "href": "../periodicity/electrical_resistivity/" }
    electrical_resistivity = soup.findChild(name="a", attrs=attrs).parent.prettify()
    electrical_resistivity = re.search(":\s*(?:about)?\s*(\d+(?:\.\d+)?)", electrical_resistivity)
    if electrical_resistivity is not None and electrical_resistivity.group(1) not in ["no", "&gt;"]:
      atom.electrical_resistivity = float(electrical_resistivity.group(1)) * 1e-8 * pq.ohm * pq.m

    results[str(atom.symbol)] = atom
    
    if atom.symbol in orbital_radii:
      au = a0("A") * pq.angstrom 
      results[str(atom.symbol)].orbital_radii = tuple([u * au for u in orbital_radii[atom.symbol]])
    if atom.symbol in pettifor_numbers:
      results[str(atom.symbol)].pettifor = pettifor_numbers[atom.symbol]


  with open(filename, "w") as file:
    file.write("\"\"\" Definition of the elements. \"\"\"\n")
    file.write("\nfrom numpy import array\n")
    file.write("\nfrom quantities import *\n")
    file.write("\nfrom . import Element\n")
    file.write("\n__dir__ = ['elements', 'symbols']\n")
    file.write("\nelements = " + repr(results) + "\n")
    keys = []
    for n in range(1, len(results)):
      for key, value in results.items():
        if value.atomic_number == n: keys.append(str(key))
    file.write("\nsymbols = {0}\n".format(keys))

    


     

