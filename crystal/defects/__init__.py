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

""" Point-defect helper functions. """
__docformat__ = "restructuredtext en"
__all__ = [ 'inequivalent_sites', 'vacancy', 'substitution', 'charged_states', \
            'band_filling', 'potential_alignment', 'charge_corrections', \
            'magmom', 'low_spin_states', 'high_spin_states', 'magname', \
            'ExtractSingle', 'ExtractMaterial' ]

from extract import Single as ExtractSingle, Material as ExtractMaterial

def symmetrically_inequivalent_sites(lattice, type):
  """ Yields sites occupied by type which are inequivalent according to symmetry operations. 
  
      When creating a vacancy on, say, "O", or a substitution of "Al" by "Mg",
      there may be more than one site which qualifies. We want to iterate over
      those sites which are inequivalent only, so that only the minimum number
      of operations are performed. 

      :note:
        lattice sites can be defined as occupiable by more than one atomic type\:
        lattice.site.type[i] = ["Al", "Mg"]. These sites will be counted if
        type in lattice.site.type, where type is the input parameter.
 
      :Parameters:
          lattice : `pylada.crystal.Lattice`
            Lattice for which to find equivalent sites.
          type : str 
            Atomic specie for which to find inequivalent sites. 

      :return: indices of inequivalent sites.
  """
  from numpy.linalg import inv, norm
  from .. import fold_vector

  # all sites with occupation "type". 
  sites = [site for site in lattice.sites if type in site.type]
  site_indices = [i for i,site in enumerate(lattice.sites) if type in site.type]

  # inverse cell.
  invcell = inv(lattice.cell)
  # loop over all site with type occupation.
  i = 0
  while i < len(sites):
    # iterates over symmetry operations.
    for op in lattice.space_group:
      pos = op(site.pos)
      # finds index of transformed position, using translation quivalents.
      for t, other in enumerate(sites):
        if norm(fold_vector(pos, lattice.cell, invcell)) < 1e-12:
          print t
          break
      # removes equivalent site and index from lists if necessary
      if t != i and t < len(sites): 
        sites.pop(t)
        site_indices.pop(t)
    i += 1
  return site_indices


def first_shell(structure, pos, tolerance=0.25):
  """ Iterates though first neighbor shell. """
  from .. import Neighbors

  neighs = [n for n in Neighbors(structure, 12, pos)]
  d = neighs[0].distance
  return [n for n in neighs if abs(n.distance - d) < tolerance * d]

def coordination_number(structure, pos, tolerance=0.25):
  """ Returns coordination number of given position in structure. """
  return len(first_shell(structure, pos, tolerance))
  
def coordination_inequivalent_sites(lattice, type, tolerance=0.25):
  """ Yields sites occupied by type which are inequivalent according to their coordination number. 
  
      When creating a vacancy on, say, "O", or a substitution of "Al" by "Mg",
      there may be more than one site which qualifies. We want to iterate over
      those sites which are inequivalent only, so that only the minimum number
      of operations are performed. In this case, inequivalent means different
      number of first neighbors.

      :note:
        lattice sites can be defined as occupiable by more than one atomic type\:
        lattice.site.type[i] = ["Al", "Mg"]. These sites will be counted if
        type in lattice.site.type, where type is the input parameter.
 
      :Parameters:
          lattice : `pylada.crystal.Lattice`
            Lattice for which to find equivalent sites.
          type : str 
            Atomic specie for which to find inequivalent sites. 

      :return: indices of inequivalent sites.
  """
  # all sites with occupation "type". 
  sites = [(i, site) for i, site in enumerate(lattice.sites) if type in site.type]

  indices = []
  coords  = set()
  for i, site in sites:
    coord = coordination_number(lattice.to_structure(), site.pos, tolerance)
    if coord not in coords:
      indices.append(i)
      coords.add(coord)
  return indices


def non_interstitials(structure, indices, mods):
  """ Yields substitutions for given indices to structure. """
  from copy import deepcopy
  from re import compile
  if not hasattr(indices, '__len__'): indices = [indices]

  cation_regex  = compile(r'^\s*cations?\s*$')
  vacancy_regex = compile(r'^\s*vacanc(?:y|ies)\s*$')
  specie_regex  = compile(r'^\s*[A-Z][a-z]?(?:\d+)?\s*$')

  # modify input to something which makes sense and check it.
  if mods is None: all_mods = [None]
  elif isinstance(mods, str): all_mods = [mods]
  else: all_mods = mods
  mods = []
  whatthehell = []
  for type in all_mods:
    if type is None: mods.append(None)
    elif cation_regex.match(type.lower()) is not None:  mods.extend(_cationic_species(structure)) 
    elif vacancy_regex.match(type.lower()) is not None: mods.append(None)
    elif specie_regex.match(type) is not None: mods.append(type)
    else: whatthehell.append(type)
  assert len(whatthehell) == 0,\
         ValueError('Cannot understand following specie types: {0}.'.format(whatthehell))
  mods = list(set(mods))

  # loop over atoms to modify.
  for i, j in enumerate(indices):
    assert j < len(structure.atoms), RuntimeError("Site index not found.")
    input_atom = deepcopy(structure.atoms[j])
    input_atom.index = j
    # loop over atoms to modifications.
    for modif in mods:
      result = deepcopy(structure)
      if modif == input_atom.type: continue # can't substitute with self.
      if modif is None: 
        result.atoms.pop(j)
        name = "vacancy_{0}".format(input_atom.type)
        output_atom = deepcopy(input_atom)
        output_atom.type = 'None'
      else: 
        result.atoms[j].type = modif
        output_atom = deepcopy(input_atom)
        output_atom.type = modif
        name = "{0}_on_{1}".format(output_atom.type, input_atom.type)
      if len(indices) > 1: name += "/site_{0}".format(i)
      result.name = name
      yield result, output_atom, input_atom.type

def inequiv_non_interstitials(structure, lattice, type, mods, do_coords = True, tolerance=0.25):
  """ Loop over inequivalent non-interstitials. """
  if do_coords: type = type.split()[0]
  inequivs = coordination_inequivalent_sites(lattice, type, tolerance) if do_coords \
             else symmetrically_inequivalent_sites(lattice, type)
  indices = []
  for i in inequivs:
    # finds first qualifying atom
    for which, atom in enumerate(structure.atoms):
      if atom.site == i: break
    indices.append(which)
  for result in non_interstitials(structure, indices, mods): yield result


def interstitials(structure, lattice, interstitials):
  """ Yields interstitial. """
  from copy import deepcopy
  from numpy import dot
  for desc in interstitials:
    assert hasattr(desc, "__iter__"),\
           ValueError("For interstitials, desc should be a sequence: {0}".format(desc))
    assert len([u for u in desc]) == 3,\
           ValueError("For interstitials, desc should be a sequence of length 3: {0}".format(desc))
    type, position, name = tuple([u for u in desc])
    result = deepcopy(structure)
    result.add_atom = dot(lattice.cell, position), type
    result.name = "{0}_interstitial_{1}".format(type, name)
    defect = deepcopy(result.atoms[-1])
    defect.index = -1
    yield result, defect, 'None'

def _cationic_species(structure):
  """ Returns list of cationic species. """
  return list(set([a.type for a in structure.atoms if a.type not in ['O', 'S', 'Se', 'Te']]))

def iterdefects(structure, lattice, defects, tolerance=0.25):
  """ Iterates over all defects for any number of types and modifications. """
  from re import compile

  cation_regex       = compile(r'^\s*cations?\s*$')
  cation_id_regex    = compile(r'^\s*cations?(\d+)\s*$')
  cation_coord_regex = compile(r'^\s*cations?\s+coord\s*$')
  interstitial_regex = compile(r'^\s*interstitials?\s*$')
  type_regex         = compile(r'^\s*(?:vacanc(?:y|ies)|[A-Z][a-z]?(?:\d+)?)(?:\s+coord)?\s*$')
  already_cations    = False

  for key, value in defects.items():
    if key is None: keys = [None]
    elif interstitial_regex.match(key.lower()) is not None: keys = [None]
    elif cation_regex.match(key.lower()) is not None:
      assert already_cations == False, ValueError('Can only have one cation tag.')
      already_cations = True
      keys = _cationic_species(structure)
    elif cation_id_regex.match(key.lower()) is not None:
      assert already_cations == False, ValueError('Can only have one cation tag.')
      already_cations = True
      d = int(cation_id_regex.match(key.lower()).group(1))
      keys = ['{0}{1}'.format(k, d) for k in _cationic_species(structure)]
    elif cation_coord_regex.match(key.lower()) is not None:
      assert already_cations == False, ValueError('Can only have one cation tag.')
      already_cations = True
      keys = ['{0} coord'.format(k) for k in _cationic_species(structure)]
    else: keys = [key]
    for type in keys:
      assert type is None or type_regex.match(type) is not None,\
             ValueError("Cannot understand type {0}.".format(type))
      for result in any_defect(structure, lattice, type, value, tolerance): yield result

def any_defect(structure, lattice, type, subs, tolerance=0.25):
  """ Yields point-defects of a given type and different modifications. 
  
      Loops over all equivalent point-defects.

      :Parameters:
        structure : `pylada.crystal.Structure`
          structure on which to operate
        lattice : `pylada.crystal.Lattice`
          back-bone lattice of the structure.
        type : str or None or sequence
          type of atoms for which to create substitution.
          If None, will create a vacancy.
          If ``subs`` is None, then will create a vacancy. In that case, type
          should be a sequence describing the interstitials:

          >>> type = [ "Li", (0,0,0), "16c" ],\
          >>>        [ "Li", (0.75,0.75,0.75), "32e_0.75" ] 
           
          Each item in the sequence is itself a sequence where the first item is the
          specie, and the other items the positions and name of the
          interstitials for that specie. 
        subs : str or None
          substitution type. If None, will create an interstitial.

      :return: a 2-tuple consisting of:

        - the structure with a substitution.
        - the substituted atom in the structure above. The atom is given an
          additional attribute, C{index}, referring to list of atoms in the
          structure.
  """
  from re import compile

  specie_regex = compile(r'^\s*[A-Z][a-z]?\s*$')
  id_regex     = compile(r'^\s*([A-Z][a-z]?)(\d+)\s*$')
  coord_regex  = compile(r'^\s*([A-Z][a-z]?)\s+coord\s*$')

  # Interstitials.
  if hasattr(type, 'rstrip'): type = type.rstrip()
  if hasattr(type, 'lstrip'): type = type.lstrip()
  if type is None or type.lower() in ['interstitial', 'interstitials', 'none']: 
    for result in interstitials(structure, lattice, subs): yield result
  # Old: looking for specific atoms.
  elif id_regex.match(type) is not None:
    # looks for atom to modify
    found = id_regex.match(type)
    type, index = found.group(1), int(found.group(2)) 
    if index < 1: index = 1
    for i, site in enumerate(lattice.sites):
      if type not in site.type: continue
      index -= 1
      if index == 0: break
    assert index == 0, ValueError("Could not find {0}.".format(type))
    for index, atom in enumerate(structure.atoms):
      if atom.site == i: break
    assert atom.site == i, ValueError('Could not find atomic-site.')
    for result in non_interstitials(structure, index, subs): yield result
  # O, Mn ... but not O1: looking for symmetrically inequivalent sites.
  elif specie_regex.match(type) is not None: 
    for result in inequiv_non_interstitials(structure, lattice, type, subs, False, tolerance): yield result
  elif coord_regex.match(type) is not None: 
    for result in inequiv_non_interstitials(structure, lattice, type, subs, True, tolerance): yield result
  else: raise ValueError("Don't understand defect type {0}".format(type))

def charged_states(species, A, B):
  """ Loops over charged systems. 

      Charged states are given as A on B, where A and B are either None or an
      atomic-specie. If B is None, this indicates that A should be an
      interstitial. If A is None, than the defect is a vacancy.
      The charge states are as follows:
      
      - vacancy: between 0 and -B.oxidation.
      - interstitial: between 0 and A.oxidation.
      - substitution: between the maximum and minimum values of 
        A.oxidation - B.oxidation, -B.oxidation, 0.

      :Parameters:
        species : `pylada.vasp.specie.Specie`
          A dictionary containing the description of the atomic species.
        A : None, or str
          If None, indicates that the charge states of an interstial are
          requested. If a string, it should be an atomic symbol with an entry
          in `species`. The defect is then either a vacancy or a substitution,
          depending on `B`. 
        B : None, or str
          If None, indicates that the charge states of a vacancy are
          requested. If a string, it should be an atomic symbol with an entry
          in `species`. The defect is then either an interstitial or a
          substitution, depending on `A`. 

      :return: Yields a 2-tuple:

        - Number of electrons to add to the system (not charge). 
        - a suggested name for the charge state calculation.
  """
  if A == 'None': A = None
  if B == 'None': B = None
  assert A is not None or B is not None, ValueError("Both A and B cannot be None")

  if A is None:   # vacancy! Charge states are in 0 to -B.oxidation.
    B = species[B]
    max_charge = -B.oxidation if hasattr(B, "oxidation") else 0
    min_charge = 0
  elif B is None: # interstial! Charge states are in 0, A.oxidation.
    A = species[A[0]]
    max_charge = A.oxidation if hasattr(A, "oxidation") else 0
    min_charge = 0
  else:           # substitution! Charge states are difference of A and B.
    A, B = species[A], species[B]
    Aox = A.oxidation if hasattr(A, "oxidation") else 0
    Box = B.oxidation if hasattr(B, "oxidation") else 0
    max_charge = max(Aox - Box, -Box, 0)
    min_charge = min(Aox - Box, -Box, 0)
    
  if max_charge < min_charge: max_charge, min_charge = min_charge, max_charge

  for charge in range(min_charge, max_charge+1):
    # directory
    if   charge == 0:   oxdir = "charge_neutral"
    elif charge > 0:    oxdir = "charge_" + str(charge) 
    elif charge < 0:    oxdir = "charge_" + str(charge) 
    yield -charge, oxdir


def band_filling(defect, host, vbm=None, cbm=None, potal=None, **kwargs):
  """ Returns band-filling corrrection. 

      :Parameters: 

        defect 
          An output extraction object as returned by the vasp functional when
          computing the defect of interest.
        host 
          An output extraction object as returned by the vasp functional when
          computing the host matrix.
        vbm, cbm, potal
          float or None, in eV
          Default None.
        kwargs 
          Parameters are passed on to potential alignment calculations.
         
      :return: Band-filling correction in eV.

      Accounts for Moss-Burnstein band-filling effects in the case of shallow
      donors and acceptors.
      The result of this call should be added to the energy of the point-defect.
  """
  from numpy import sum, multiply, newaxis
  from quantities import eV

  potal = potential_alignment(defect, host, **kwargs) if potal == None else potal*eV

  cbm = (host.cbm if cbm == None else cbm*eV) + potal

  if defect.eigenvalues.ndim == 3:
    dummy = multiply(defect.eigenvalues-cbm, defect.multiplicity[newaxis,:,newaxis])
    dummy = multiply(dummy, defect.occupations)
  elif defect.eigenvalues.ndim == 2:
    dummy = multiply(defect.eigenvalues-cbm, defect.multiplicity[:, newaxis])
    dummy = multiply(dummy, defect.occupations)
  result = -sum(dummy[defect.eigenvalues > cbm])

  vbm = (host.vbm if vbm == None else vbm*eV) + potal

  if defect.eigenvalues.ndim == 3:
    dummy = multiply(vbm-defect.eigenvalues, defect.multiplicity[newaxis,:,newaxis])
    dummy = multiply(dummy, 1e0-defect.occupations)
  elif defect.eigenvalues.ndim == 2:
    dummy = multiply(vbm-defect.eigenvalues, defect.multiplicity[:, newaxis])
    dummy = multiply(dummy, 2e0-defect.occupations)
  result -= sum(dummy[defect.eigenvalues < vbm])

  return -result.rescale(eV) / sum(defect.multiplicity)
  
def explore_defect(defect, host, **kwargs):
  """ Diagnostic tool to determine defect from defect calculation and host. 
  
      :Parameters:
        defect : `pylada.vasp.ExtractDFT`
          Extraction object for the vasp calculation of the defect structure.
          The defect structure should be a supercell of the host. Its unit cell
          must be an exact multiple of the host unit cell. Atoms may have moved
          around, however.
        host : `pylada.vasp.ExtractDFT`
          Extraction object for the vasp calculation of the host structure.
        kwargs 
          Passed on to `reindex_sites`.

      :return: 
        Dictionary containing three items:

        - 'vacancy': a list of atoms from the host *missing* in the defect
          structure. The position of these atoms correspond to the missing atom
          in the supercell (not the translational equivalent of the unit-cell).
        - 'substitution': list of indices referring to atoms in the defect
          structure with substituted types (w.r.t. the host).
        - 'intersititial': list of indices referring to atoms in the defect
          structure with no counterpart in the host structure.
     
      :note: The results may be incorrect if the defects incur too much relaxation. 
  """
  from copy import deepcopy
  from pylada.crystal.defects import reindex_sites
  from pylada.crystal import structure_to_lattice
  
  dstr = defect.structure.copy()
  hstr = host.structure
  # modified by Haowei: using the p1 structure for reindexing and potential_alignment 
  hlat_p1 = structure_to_lattice(hstr, primitive=False)
  reindex_sites(dstr, hlat_p1, **kwargs)

  result = {'interstitial': [], 'substitution': [], 'vacancy': []}
  # looks for intersitials and substitutionals.
  for i, atom in enumerate(dstr.atoms):
    if atom.site == -1: 
      result['interstitial'].append(i)
    elif atom.type != hstr.atoms[atom.site].type: 
      result['substitution'].append(i)

  # looks for vacancies.
  filled = hlat_p1.to_structure(dstr.cell)
  reindex_sites(filled, dstr, **kwargs)
  for atom in filled.atoms:
    if atom.site != -1: continue
    result['vacancy'].append(deepcopy(atom))
  return result

def potential_alignment(defect, host, maxdiff=None, first_shell=False, tolerance=0.25):
  """ Returns potential alignment correction. 

      :Parameters:

        defect 
          An output extraction object as returned by the vasp functional when
          computing the defect of interest.
        host 
          An output extraction object as returned by the vasp functional when
          computing the host matrix.
        maxdiff : float or None, in eV(?)
          Maximum difference between the electrostatic potential of an atom and
          the equivalent host electrostatic potential beyond which that atom is
          considered pertubed by the defect.
          If None or negative, then differences in electrostatice potentials
          are not considered.
          Default 0.5.
        first_shell : bool 
          If true then removes from potential alignment the first neighbor of
          defect atoms.
          Default False.
        tolerance
          Passed on to `reindex_sites`.
          Default 0.25.  
      :return: The potential alignment in eV (without charge factor).

      Returns average difference of the electrostatic potential of the
      unperturbed atoms in the defect structure with respect to the host.
      *Perturbed* atoms are those flagged as defect by `explore_defect`, their
      first coordination shell if ''first_shell'' is true, and atoms for which
      the electrostatic potential differ to far from the average electrostatic
      potential for each lattice site (parameterized by maxdiff).

  """
  from itertools import chain
  from numpy import abs, array, mean, any
  from quantities import eV
  from . import reindex_sites, first_shell as ffirst_shell
  from pylada.crystal import structure_to_lattice

  dstr = defect.structure
  hstr = host.structure
  # modified by Haowei: uring the p1 structure for potential alignment
  # this is necessary for two reasons:
  # 1, the host unit cell may be a *supercell* due to the magnetic structure  
  # 2, in the conventional defect calculations, we run the calculation for host using the same supercell as defect calculations
  #    but this should not be a problem if the calculations are converged very well
  hlat_p1 = structure_to_lattice(hstr, primitive=False)
  reindex_sites(dstr, hlat_p1, tolerance=tolerance)
  defects = explore_defect(defect, host, tolerance=tolerance)
  acceptable = [True for a in dstr.atoms]
  # make interstitials and substitutionals unaceptable.
  for i in chain(defects['interstitial'], defects['substitution']):
    acceptable[i] = False
    if first_shell:
      for n in ffirst_shell(dstr, dstr.atoms[i].pos, tolerance=tolerance):
        acceptable[n.index] = False
  # makes vacancies unacceptable.
  if first_shell:
    for atom in defects['vacancy']:
      for n in ffirst_shell(dstr, atom.pos, tolerance=tolerance):
        acceptable[n.index] = False

  # make a deepcopy for backup 
  raw_acceptable = list(acceptable)
  if maxdiff != None and maxdiff > 0.0:
    # directly compare the atomic site between the host and defect cell/supercell
    diff_dh = [ (0.0 * eV if not ok else abs(e - host.electropot[a.site]).rescale(eV)) \
         for e, a, ok in zip(defect.electropot, dstr.atoms, acceptable) ]
  
    for ixx in range(len(acceptable)):
      if acceptable[ixx] == False: pass
      elif float(diff_dh[ixx].magnitude) > maxdiff: acceptable[ixx] = False

  if not any(acceptable):
    # if some one try to use maxdiff = 0.0000000001, @&#(@&#(#@^@
    print "WARNING: maxdiff is too small! Jump to maxdiff=None"
    # return to the default one, which accept all the atomic sites except the defect sites
    acceptable = list(raw_acceptable) 

  iterable = zip(defect.electropot, dstr.atoms, acceptable)

  return mean([ (e - host.electropot[a.site]).rescale(eV).magnitude\
                for e, a, ok in iterable if ok ]) * eV


def third_order_charge_correction(structure, charge = None, n = 30, epsilon = 1.0, **kwargs):
  """ Returns energy of third order charge correction. 
  
      :Parameters: 
        structure : `pylada.crystal.Structure`
          Defect supercell, with cartesian positions in angstrom.
        n 
          precision. Higher better.
        charge 
          If no units are given, defaults to elementary charge. If None,
          defaults to 1 elementary charge.
        epsilon 
          Static dielectrict constant of the host. Most likely in atomic units
          (e.g. dimensionless), but I'm not sure.
      
      Taken as is from `Lany and Zunger, PRB 78, 235104 (2008)`__.
      Always outputs as eV. Not sure what the units of some of these quantities are. 

      .. __:  http://dx.doi.org/10.1103/PhysRevB.78.235104

      :return: third order correction  to the energy in eV. Should be *added* to total energy.
  """
  from quantities import elementary_charge, eV, pi, angstrom
  from ...physics import a0, Ry
  from .._crystal import third_order

  if charge is None: charge = 1e0
  elif charge == 0: return 0e0 * eV
  if hasattr(charge, "units"):  charge  = float(charge.rescale(elementary_charge))
  if hasattr(epsilon, "units"): epsilon = float(epsilon.simplified)
  cell = (structure.cell*structure.scale*angstrom).rescale(a0)
  return third_order(cell, n) * (4e0*pi/3e0) * Ry.rescale(eV) * charge * charge \
         * (1e0 - 1e0/epsilon) / epsilon
         

def first_order_charge_correction(structure, charge=None, epsilon=1e0, cutoff=20.0, **kwargs):
  """ First order charge correction of +1 charge in given supercell. 
  
      Units in this function are either handled by the module Quantities, or
      defaults to Angstroems and elementary charges.

      :Parameters:
        structure : `pylada.crystal.Structure`
          Defect supercell, with cartesian positions in angstrom.
        charge 
          Charge of the point-defect. Defaults to 1e0 elementary charge. If no
          units are attached, expects units of elementary charges.
        epsilon 
          dimensionless relative permittivity.
        cutoff 
          Ewald cutoff parameter.

      :return: Electrostatic energy in eV.
  """
  from quantities import elementary_charge, eV
  from .. import Structure
  from ...physics import Ry
  try: from ...pcm import Clj 
  except ImportError as e:
    from warnings import warn
    warn(ImportWarning("Could not import Point-Charge Model package (pcm). \n"\
                       "Cannot compute first order charge correction.\n"\
                       "Please compile Pylada with pcm enabled.\n"))
    return 

  if charge is None: charge = 1
  elif charge == 0: return 0e0 * eV
  if hasattr(charge, "units"): charge = float(charge.rescale(elementary_charge))

  clj = Clj()
  clj.charges["A"] = charge
  clj.ewald_cutoff = cutoff * Ry

  struc = Structure()
  struc.cell = structure.cell
  struc.scale = structure.scale
  struc.add_atom = ((0e0,0,0), "A")

  result = clj.ewald(struc).energy / epsilon
  return -result * eV

def charge_corrections(structure, **kwargs):
  """ Electrostatic charge correction (first and third order). 

      Computes first and third order charge corrections according to `Lany
      and Zunger, PRB 78, 235104 (2008)`__. Calculations are
      done for the correct charge of the system and a static dielectric
      constant epsilon=1. For other static dielectric constants, use:

      >>> correction = output.charge_corrections / epsilon

      For conventional and unit-cells of Ga2MnO4 spinels, the charge
      corrections are converged to roughly 1e-5 eV (for singly charged).

      .. __:  http://dx.doi.org/10.1103/PhysRevB.78.235104
  """
  return   first_order_charge_correction(structure, **kwargs) \
         - third_order_charge_correction(structure, **kwargs) \

def magnetic_neighborhood(structure, defect, species):
   """ Finds magnetic neighberhood of a defect. 
   
       If the defect is a substitution with a magnetic atom, then the
       neighberhood is the defect alone. Otherwise, the neighberhood extends to
       magnetic first neighbors. An atomic specie is deemed magnetic if marked
       as such in `species`.

       :Parameters: 
         structure : `pylada.crystal.Structure`
           The structure with the point-defect already incorporated.
         defect : `pylada.crystal.Atom`
           The point defect, to which and *index* attribute is given denoting
           the index of the atom in the original supercell structure (without
           point-defect).
         species : dict of `pylada.vasp.species.Specie`
           A dictionary defining the atomic species.

       :return: indices of the neighboring atoms in the point-defect `structure`.
   """
   from numpy.linalg import norm
   from . import Neighbors

   # checks if substitution with a magnetic defect.
   if hasattr(defect, "index") and defect.index < len(structure.atoms):
     atom = structure.atoms[defect.index]
     if species[atom.type].magnetic and norm(defect.pos - atom.pos) < 1e-12:
       return [defect.index]
   # now finds first neighbors. 12 is the highest coordination number, so
   # this should include the first shell.
   neighbors = [n for n in Neighbors(structure, 12, defect.pos)]
   # only take the first shell and keep indices (to atom in structure) only.
   neighbors = [n.index for n in neighbors if n.distance < neighbors[0].distance + 1e-1]
   # only keep the magnetic neighborhood.
   return [n for n in neighbors if species[structure.atoms[n].type].magnetic]

def equiv_bins(n, N):
  """ Generator over ways to fill N equivalent bins with n equivalent balls. """
  from itertools import chain
  from numpy import array
  assert N > 0
  if N == 1: yield [n]; return
  if n == 0: yield [0 for x in range(N)]
  for u in xrange(n, 0, -1):
    for f in  equiv_bins(n-u, N-1):
      result = array([x for x in chain([u], f)])
      if all(result[0:-1]-result[1:] >= 0): yield result

def inequiv_bins(n, N):
  """ Generator over ways to fill N inequivalent bins with n equivalent balls. """
  from itertools import permutations
  for u in equiv_bins(n, N):
    u = [v for v in u]
    history = []
    for perm in permutations(u, len(u)):
      seen = False
      for other in history:
        same = not any( p != o for p, o in zip(perm, other) )
        if same: seen = True; break
      if not seen: history.append(perm); yield [x for x in perm]

def electron_bins(n, atomic_types):
  """ Loops over electron bins. """
  from itertools import product 
  from numpy import zeros, array
  # creates a dictionary where every type is associated with a list of indices into atomic_types.
  Ns = {}
  for type in set(atomic_types):
    Ns[type] = [i for i,u in enumerate(atomic_types) if u == type]
  # Distributes electrons over inequivalent atomic types.
  for over_types in inequiv_bins(n, len(Ns.keys())):
    # now distributes electrons over each type independently.
    iterables = [ equiv_bins(v, len(Ns[type])) for v, type in zip(over_types, Ns.keys()) ] 
    for nelecs in product(*iterables):
      # creates a vector where indices run as in atomic_types argument.
      result = zeros((len(atomic_types),), dtype="float64")
      for v, (type, value) in zip(nelecs, Ns.items()): result[value] = array(v)
      yield result

def magmom(indices, moments, nbatoms):
  """ Yields a magmom string from knowledge of which moments are non-zero. """
  s = [0 for i in range(nbatoms)]
  for i, m in zip(indices, moments): s[i] = m
  compact = [[1, s[0]]]
  for m in s[1:]:
    if abs(compact[-1][1] - m) < 1e-12: compact[-1][0] += 1
    else: compact.append( [1, m] )
    
  string = ""
  for n, m in compact:
    if n > 1: string +=" {0}*{1}".format(n, m)
    elif n == 1: string += " {0}".format(m)
    assert n != 0
  return string

def electron_counting(structure, defect, species, extrae):
  """ Enumerate though number of electron in point-defect magnetic neighberhood. 

      Generator over the number of electrons of each atom in the magnetic
      neighberhood of a point defect with `extrae` electrons. If there are no
      magnetic neighborhood, then `magmom` is set
      to None and the total magnetic moment to 0 (e.g. lets VASP figure it out).
      Performs a sanity check on integers to make sure things are correct.

      :Parameters:
        structure : `pylada.crystal.Structure`
          Structure with point-defect already inserted.
        defect : `pylada.crystal.Atom`
          Atom making up the point-defect.
          In addition, it should have an *index* attribute denoting the defect 
        species : dict of `pylada.vasp.species.Specie`
          Dictionary containing details of the atomic species.
        extrae
          Number of extra electrons to add/remove.

      :return: yields (indices, electrons) where indices is a list of indices
        to the atom in the neighberhood, and electrons is a corresponding list of
        elctrons.
  """
  from numpy import array
  from ...physics import Z
  indices = magnetic_neighborhood(structure, defect, species)

  # no magnetic neighborhood.
  if len(indices) == 0: 
    yield None, None
    return

  # has magnetic neighberhood from here on.
  atoms = [structure.atoms[i] for i in indices]
  types = [a.type for a in atoms]
  nelecs = array([species[type].valence - species[type].oxidation for type in types])

  # loop over different electron distributions.
  for tote in electron_bins(abs(extrae), types):
    # total number of electrons on each atom.
    if extrae < 0:   tote = nelecs - tote
    elif extrae > 0: tote += nelecs

    # sanity check. There may be more electrons than orbitals at this point.
    sane = True
    for n, type in zip(tote, types):
      if n < 0: sane = False; break;
      z = Z(type)
      if (z >= 21 and z <= 30) or (z >= 39 and z <= 48) or (z >= 57 and z <= 80):  
        if n > 10: sane = False;  break
      elif n > 8: sane = False; break

    if not sane: continue

    yield indices, tote
 

def low_spin_states(structure, defect, species, extrae, do_integer=True, do_average=True):
  """ Enumerate though low-spin-states in point-defect. 

      Generator over low-spin magnetic states of a defect with
      `extrae` electrons. The extra electrons are distributed both as integers
      and as an average. All these states are ferromagnetic. In the special
      case of a substitution with a magnetic atom, the moment is expected to go
      on the substitution alone. If there are no magnetic neighborhood, then `magmom` is set
      to None and the total magnetic moment to 0 (e.g. lets VASP figure it out).
      
      :Parameters:
        structure : `pylada.crystal.Structure`
          Structure with point-defect already inserted.
        defect : `pylada.crystal.Atom`
          Atom making up the point-defect.
          In addition, it should have an *index* attribute denoting the defect 
        species : dict of `pylada.vasp.species.Specie`
          Dictionary containing details of the atomic species.
        extrae
          Number of extra electrons to add/remove.

      :return: yields (indices, moments) where the former index the relevant
               atoms in `structure` and latter are their respective moments.
  """
  from numpy import array, abs, all

  history = []
  def check_history(*args):
    for i, t in history:
      if all(abs(i-args[0]) < 1e-12) and all(abs(t-args[1]) < 1e-12):
        return False
    history.append(args)
    return True


  if do_integer: 
    for indices, tote in electron_counting(structure, defect, species, extrae):
      if tote is None: continue # non-magnetic case
      indices, moments = array(indices), array(tote) % 2
      if all(abs(moments) < 1e-12): continue # non - magnetic case
      if check_history(indices, moments): yield indices, moments
  if do_average: 
    for indices, tote in electron_counting(structure, defect, species, 0):
      if tote is None: continue # non-magnetic case
      if len(indices) < 2: continue
      indices, moments = array(indices), array(tote) % 2 + extrae / float(len(tote))
      if all(abs(moments) < 1e-12): continue # non - magnetic case
      if check_history(indices, moments): yield indices, moments


def high_spin_states(structure, defect, species, extrae, do_integer=True, do_average=True):
  """ Enumerate though high-spin-states in point-defect. 

      Generator over high-spin magnetic states of a defect with
      `extrae` electrons. The extra electrons are distributed both as integers
      and as an average. All these states are ferromagnetic. In the special
      case of a substitution with a magnetic atom, the moment is expected to go
      n the substitution alone. If there are no magnetic neighborhood, then
      `magmom` is set to None and the total magnetic moment to 0 (e.g. lets
      VASP figure it out).

      :Parameters:
        structure : `pylada.crystal.Structure`
          Structure with point-defect already inserted.
        defect : `pylada.crystal.Atom`
          Atom making up the point-defect.
          In addition, it should have an *index* attribute denoting the defect 
        species : dict of `pylada.vasp.species.Specie`
          Dictionary containing details of the atomic species.
        extrae 
          Number of extra electrons to add/remove.

      :return: yields (indices, moments) where the former index the relevant
               atoms in `structure` and latter are their respective moments.
  """
  from numpy import array, abs, all

  def is_d(t): 
    """ Determines whether an atomic specie is transition metal. """
    from ...physics import Z
    z = Z(t)
    return (z >= 21 and z <= 30) or (z >= 39 and z <= 48) or (z >= 57 and z <= 80) 

  def determine_moments(arg, ts): 
    """ Computes spin state from number of electrons. """
    f = lambda n, t: (n if n < 6 else 10-n) if is_d(t) else (n if n < 5 else 8-n)
    return [f(n,t) for n, t in zip(arg, ts)]

  history = []
  def check_history(*args):
    for i, t in history:
      if all(abs(i-args[0]) < 1e-12) and all(abs(t-args[1]) < 1e-12):
        return False
    history.append(args)
    return True
  
  if do_integer: 
    for indices, tote in electron_counting(structure, defect, species, extrae):
      if tote is None: continue # non-magnetic case
      
      types = [structure.atoms[i].type for i in indices]
      indices, moments = array(indices), array(determine_moments(tote, types))
      if all(moments == 0): continue # non - magnetic case
      if check_history(indices, moments):  yield indices, moments

  if do_average: 
    for indices, tote in electron_counting(structure, defect, species, 0):
      if tote is None: continue # non-magnetic case
      if len(indices) < 2: continue

      types = [structure.atoms[i].type for i in indices]
      indices = array(indices)
      moments = array(determine_moments(tote, types)) + float(extrae) / float(len(types))
      if all(abs(moments) < 1e-12): continue # non - magnetic case
      if check_history(indices, moments):  yield indices, moments

def reindex_sites(structure, lattice, tolerance=0.5):
  """ Reindexes atoms of structure according to lattice sites.
  
      Expects that the structure is an exact supercell of the lattice, as far
      cell vectors are concerned. The atoms, however, may have moved around a
      bit. To get an index, an atom must be clearly closer to one ideal lattice
      site than to any other, within a given tolerance (in units of `structure.scale`?).
  """
  from .. import Neighbors
  if hasattr(lattice, 'to_lattice'): lattice = lattice.to_lattice()
  lattice = lattice.to_structure(structure.cell)
  for atom in structure.atoms:
    neighs = [n for n in Neighbors(lattice, 2, atom.pos)]
    assert abs(neighs[1].distance) > 1e-12,\
           RuntimeError('Found two sites occupying the same position.')
    if neighs[0].distance > tolerance: atom.site = -1
    else: atom.site = lattice.atoms[neighs[0].index].site

def magname(moments, prefix=None, suffix=None):
  """ Construct name for magnetic moments. """
  if len(moments) == 0: return "paramagnetic"
  string = str(moments[0])
  for m in moments[1:]: string += "_" + str(m)
  if prefix is not None: string = prefix + "_" + string
  if suffix is not None: string += "_" + suffix
  return string
