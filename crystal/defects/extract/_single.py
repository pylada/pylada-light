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

"""" Extraction object for single point-defect calculations. """
__docformat__ = "restructuredtext en"
__all__ = ['Single']
from ....opt.decorators import make_cached

class _ChargedStateNavigation(object):
  """ Base class containing charged state navigation methods.
  
  
      Navigates around a single defect. A *single* defect can be characterized
      by a number of different charge states. Each charge state itself can be
      describe by a number of (presumably) magnetic states. However, this last
      stage is optimized out and will not appear explicitely in the interface
      of this object.
  """
  def __init__(self, extract, epsilon = 1e0, host = None, pa_kwargs=None):
    """ Initializes an enthalpy function. """
    super(_ChargedStateNavigation, self).__init__()
    # extraction object.
    self.extract = extract.copy(unix_re=False)
    if self.extract.excludes is None: self.extract.excludes = [".*relax_*"]
    else: self.extract.excludes.append(".*relax_*$")

    self._host = host
    """ Host extraction object. 

        If None, will determine it from the directory structure.
    """
    self.epsilon = epsilon
    """ Dimensionless dielectric constant. """
    self.pa_kwargs = pa_kwargs 
    """ Potential alignment parameter. """
    if self.pa_kwargs is None: self.pa_kwargs = {}

  @property
  def epsilon(self):
    """ Dimensionless dielectric constant of the host material. """
    return self._epsilon
  @epsilon.setter
  def epsilon(self, value): 
    self._epsilon = value
    self.uncache()

  @property
  def rootdir(self):
    """ Root directory of defects. """
    return self.extract.rootdir

  def _all_jobs(self):
    """ Loops over all jobs in special way. """
    for child in self.extract.children: # Each child is a different charge state.
      for job in child.itervalues(): yield job

  @property
  @make_cached
  def _charge_correction(self):
    """ Returns the charge corrections.
    
        Tries and minimizes the number of calculations by checking if performed
        in same cell.
    """
    from numpy.linalg import inv, det
    from numpy import array
    from quantities import eV
    result, cells  = [], []
    # loops of all jobs.
    for job in self._all_jobs():
      cell = job.structure.cell
      invcell = inv(cell)
      found = None
      # looks if already exists.
      for i, other in enumerate(cells):
        rotmat = other * cell
        d = abs(det(rotmat))
        if abs(d - 1e0) < 1e-8: continue
        invrotmat = inv(rotmat)
        if all( abs(rotmat.T - invrotmat) < 1e-8 ): found = i; break
      if found is None: 
        from .. import charge_corrections
        c = charge_corrections( job.structure, charge=job.charge, 
                                epsilon=self.epsilon, n=40, cutoff=45. )
        cells.append(inv(cell))
        result.append(c.copy())
      else: result.append(result[found])
    return array(result) * eV

  @property
  @make_cached
  def _potential_alignment(self):
    """ Potential alignments for all jobs. """
    from numpy import array
    from quantities import eV
    from .. import potential_alignment
    return array([ potential_alignment(state, self.host, **self.pa_kwargs) \
                   for state in self._all_jobs() ]) * eV

  @property
  @make_cached
  def _band_filling(self):
    """ Band-filling for all jobs. """
    from numpy import array
    from quantities import eV
    from .. import band_filling
    return array([ band_filling(state, self.host, **self.pa_kwargs) \
                   for state in self._all_jobs() ]) * eV

  @property
  @make_cached
  def _uncorrected(self):
    """ Uncorrected formation enthalpy. """
    from numpy.linalg import det
    from numpy import array
    from quantities import eV
    energies = []
    for state in self._all_jobs():
      n = int(det(state.structure.cell)/det(self.host.structure.cell) + 1.e-3) + 0.
      energies.append(state.total_energy - self.host.total_energy * n)
    return array(energies) * eV 

  @property
  def _corrected(self):
    """ Corrected formation enthalpy. """
    return   self._uncorrected \
           + self._charge_correction\
           + self._potential_alignment\
           + self._band_filling


  @property
  @make_cached
  def _charged_states(self):
    """ Yields extraction routine toward each charge states. 

        If a charge state has children, then only the lowest energy calculation
        is returned.
    """
    from os.path import basename
    from operator import itemgetter
    alles = {}
    names = [child.directory for child in self._all_jobs()]
    for n, u, c, p, b, corr in zip( names, self._uncorrected, \
                                    self._charge_correction, \
                                    self._potential_alignment, \
                                    self._band_filling,\
                                    self._corrected ):
      alles[n] = u, c, p, b, corr
    corrected = self._corrected
    charges  = [u.charge for u in self._all_jobs()]
    children  = [u for u in self._all_jobs()]
    result = []
    for charge in sorted(list(set(charges))):
      sequence = [(child, u) for child, u, c in zip(children, corrected, charges) if c == charge]
      child = sorted(sequence, key=itemgetter(1))[0][0].copy()
      child.__dict__['raw_deltaH']          = alles[child.directory][0]
      child.__dict__['charge_corrections']  = alles[child.directory][1]
      child.__dict__['potential_alignment'] = alles[child.directory][2]
      child.__dict__['band_filling']        = alles[child.directory][3]
      child.__dict__['deltaH']              = alles[child.directory][4]
      result.append(child)
    return result

  @property
  def host(self):
    """ Returns extraction object towards the host. """
    if self._host is None: 
      host = self.extract['../..' if self._is_site is None else '../../..']
      host = self.copy(excludes=[".*PointDefects"], naked_end=False)
      host.excludes.extend(host.excludes)
      lowest = sorted(child.total_energies.iteritems(), key=itemgetter(1))[0][0]
      self._host = [u for u in self.extract[lowest].itervalues()]
      assert len(self._host) == 1
      self._host = self._host[0]
    return self._host

  @property 
  def _site(self):
    """ Returns site number or None. """
    from re import match
    regex = match(r"site_(\d+)", self.extract.view.split('/')[-1])
    return int(regex.group(1)) if regex is not None else None

  @property 
  def name(self):
    """ Name of the defect. """
    return self.extract.view.split('/')[-2 if self._site is not None else -1]

  @property
  def is_vacancy(self):
    """ True if this is a vacancy. """
    from re import match
    return match("vacancy_[A-Z][a-z]?", self.name) is not None

  @property
  def is_interstitial(self):
    """ True if this is an interstitial. """
    from re import match
    return match("[A-Z][a-z]?_interstitial_\S+", self.name) is not None

  @property
  def is_substitution(self):
    """ True if this is a substitution. """
    from re import match
    return match("[A-Z][a-z]?_on_[A-Z][a-z]?", self.name) is not None

  @property
  def n(self):
    """ Number of atoms added/removed from system.
    
        This is a dictionary.
    """
    from re import match
    if self.is_vacancy:
      return {match("vacancy_([A-Z][a-z])?", self.name).group(1): -1}
    elif self.is_interstitial:
      return {match("[A-Z][a-z]?_interstitial_(\S+)", self.name).group(1): -1}
    else: 
      found = match("([A-Z][a-z])?_on_([A-Z][a-z])?", self.name)
      return {match.group(1): 1, match.group(2): -1}

  def uncache(self):
    """ Uncaches result. """
    from ....opt.decorators import uncache as opt_uncache
    opt_uncache(self)
    self.extract.uncache()
    self.host.uncache()

  def __getitem__(self, value):
    """ Returns specific charge state. """
    for job in self._all_jobs():
      if abs(job.charge - value) < 1e-12: return job
    raise KeyError('Charge state {0} not found.'.format(value))
  def __contains__(self, value):
    """ True if value is a known charge state. """
    for job in self._all_jobs():
      if abs(job.charge - value) < 1e-12: return True
    return False
  def __len__(self, value): 
    """ Number of charge states. """
    return len([0 for u in self._all_jobs()])

class Single(_ChargedStateNavigation):
  """ Extracts data for a single defect.
  
      A *single* defect includes charged states, as well as magnetic states
      which have been optimized out.
  """
  def __init__(self, extract, epsilon = 1e0, host = None, pa_kwargs=None):
    """ Initializes an enthalpy function. """
    super(Single, self).__init__(extract, epsilon, host, pa_kwargs)

  def chempot(self, mu):
    """ Computes sum of chemical potential from dictionary ``mu``. 
    
        :Param mu: Dictionary of chemical potentials. If no units, assumes eV.
        :return: Chemical potential of this defect. Value is always in eV.
    """
    from quantities import eV
    if mu is None: return 0 * eV 
    result = 0e0 * eV
    n = self.n
    for specie, value in self.n:
      assert specie in mu,\
             ValueError("Specie {0} not in input chemical potentials {1}.".format(specie, mu))
      chem = mu[specie]
      if not hasattr(chem, 'units'): chem = chem * eV
      result += value * chem
    return result.rescale(eV)

  def _lines(self):
    """ Returns lines composed by the different charge states. """
    from numpy import array
    from quantities import elementary_charge as e, eV
    lines = []
    states = set()
    for state in self._charged_states:
      assert state.charge not in states,\
             RuntimeError("Found more than one calculation for the same charge state.")
      states.add(state.charge)
      lines.append((state.deltaH.rescale(eV), state.charge))
    return lines

  def _all_intersections(self, _lines):
    """ Returns all intersection points between vbm and cbm, ordered. """
    from numpy import array
    from quantities import eV
    result = []
    for i, (b0, a0) in enumerate(_lines[:-1]):
      for b1, a1 in _lines[i+1:]: result.append( (b0 - b1) / (a1 - a0) )
    return array(sorted([array(u.rescale(eV)) for u in result])) * eV

  def lines(self):
    """ Lines forming the formation enthalpy diagram. 
    
        :return: A list of 2-tuples with the first item b and the second a (a*x+b).
    """
    from numpy import array
    from quantities import eV

    _lines = self._lines()
    intersections = list(self._all_intersections(_lines))
    intersections.append(intersections[-1]+1*eV)
    result = []
    last_point = intersections[0] - 1.*eV
    last_line = None
    for intersection in intersections:
      pos = (intersection+last_point)*0.5
      func = lambda x: x[1][0] + pos*x[1][1] 
      last_point = intersection
      i, line = min(enumerate(_lines), key=func)
      if len(result) == 0 or last_line != i:
        result.append((line[0].rescale(eV), line[1]))
        last_line = i

    return result

  def enthalpy(self, fermi, mu = None):
    """ Point-defect formation enthalpy. 
    
        :Parameters:
          fermi  
            Fermi energy with respect to the host's VBM. If without
            units, assumes eV. 
          mu : dictionary or None
            Dictionary of chemical potentials. If without units, assumes eV.
            If None, chemical potential part of the formation enthalpy is
            assumed zero.

        :return: Lowest formation enthalpy for all charged states.
    """
    from quantities import eV
    if hasattr(fermi, "rescale"): fermi = fermi.rescale(eV)
    else: fermi = fermi * eV
    return min(x[0]+fermi*x[1]+ self.chempot(mu) for x in self.lines()).rescale(eV)

  def transition(self, qi, qf): 
    """ Thermodynamic transition energy.
         
        :param qi: Initial charge state.
        :param qf: Final charge state.
        :return: transition energy in eV, with respect to VBM.

        Left-hand side of Eq. 2 in `Lany and Zunger, PRB 78, 235104 (2008)`__.

        .. __:  http://dx.doi.org/10.1103/PhysRevB.78.235104
    """
    assert qi in self.charges, ValueError("Could not find charge state {0}.".format(qi))
    assert qf in self.charges, ValueError("Could not find charge state {0}.".format(qf))
    # look for initial state.
    for istate in self._charged_states: 
      if istate.charge == qi: break
    # look for final state.
    for fstate in self._charged_states: 
      if fstate.charge == qf: break

    return (fstate.deltaH - istate.deltaH) / (qi - qf)


  @property 
  def charges(self):
    """ List of charges of all charged states. """
    return sorted([s.charge for s in self._charged_states])

  @property 
  def potal(self):
    """ Array of potential alignment values. """
    from numpy import array
    return array([a.potential_alignment for a in self._charged_states])

  @property 
  def band_filling(self):
    """ Array of band-filling corrections. """
    from numpy import array
    return array([a.band_filling for a in self._charged_states])

  @property 
  def charge_corrections(self):
    """ Array of charge corrections (first and third order both). """
    from numpy import array
    return array([a.charge_corrections for a in self._charged_states])

  @property 
  def raw_deltaH(self):
    """ Array of delta H without any correction. """
    from numpy import array
    return array([a.raw_deltaH for a in self._charged_states])

  @property
  def latex_label(self):
    """ A label in LaTex format. """
    from re import match
    if self.is_interstitial:
      site = self._site
      if site is None:
        found = match("([A-Z][a-z]?)_interstitial_(.+)$", self.name) 
        return r"{0}$^{{(i)}}_{{ \mathrm{{ {1} }} }}$"\
               .format(found.group(1), found.group(2).replace('_', r"\_"))
      else:
        found = match("([A-Z][a-z]?)_interstitial_(.+)$", self.name) 
        return r"{0}$^{{(i,{2})}}_{{ \mathrm{{ {1} }} }}$"\
               .format(found.group(1), found.group(2).replace('_', r"\_"), site)
    if self.is_substitution:
      found = match("([A-Z][a-z]?)_on_([A-Z][a-z]?)", self.name) 
      site = self._site
      if site is None:
        return r"{0}$_{{ \mathrm{{ {1} }} }}$".format(found.group(1), found.group(2))
      else:
        return r"{0}$_{{ \mathrm{{ {1} }}_{{ {2} }} }}$"\
               .format(found.group(1), found.group(2), site)
    if self.is_vacancy:
      found = match("vacancy_([A-Z][a-z]?)", self.name) 
      site = self._site
      if site is None:
        return r"$\square_{{ \mathrm{{ {0} }} }}$".format(found.group(1))
      else:
        return r"$\square_{{ \mathrm{{ {0} }}_{{{1}}} }}$".format(found.group(1), site)
  
  def __str__(self):
    """ Energy and corrections for each charge defect. """
    from operator import itemgetter
    from os.path import relpath
    from numpy import array
    from numpy.linalg import det
    from quantities import eV
    result = "{0}: \n".format(self.name)
    states = sorted(((c, c.charge) for c in self._charged_states),  key = itemgetter(1))
    for extract, charge in states:
      n = int(det(extract.structure.cell)/det(self.host.structure.cell) + 1.e-3) + 0.
      a = float(extract.raw_deltaH.rescale(eV))
      b = float(extract.charge_corrections.rescale(eV))
      c = float(extract.potential_alignment.rescale(eV))
      d = float(extract.band_filling.rescale(eV))
      e = relpath(extract.directory, extract.directory + "/../../")
      result += "  - charge {0:>3}: deltaH = {1:8.4f} + {2:8.4f} + {3:8.4f}"\
                "+ {4:8.4f} = {5:8.4f} eV # {6}.\n"\
                .format(int(charge), a, b, c, d, a+b+c+d, e)
    return result
