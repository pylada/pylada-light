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

"""" Extraction object for many point-defects, but single material. """
__docformat__ = "restructuredtext en"
__all__ = ['Material']
from ....jobs import AbstractMassExtract
from ._single import Single

class _MaterialNavigator(AbstractMassExtract):
  """ Navigates around multiple defects of a single material. """
  DefectExtractor = Single
  """ Class for extracting data from a single defect. """
  def __init__(self, path=None, epsilon = None, pa_kwargs=None, **kwargs):
    """ Initializes an enthalpy function. """
    from ....vasp import MassExtract as VaspMassExtract

    # possible customization of mass defect extration object.
    MassExtractor = kwargs.pop("MassExtractor", VaspMassExtract)
    """ Object type for mass extraction. """
    # possible customization of single defect extration objects.
    self.__dict__["DefectExtractor"] \
        = kwargs.pop("DefectExtractor", _MaterialNavigator.DefectExtractor)

    AbstractMassExtract.__init__(self, **kwargs)

    self.massextract = MassExtractor(path, unix_re=False, excludes=[".*relax_*"])
    """ Mass extraction object from which all results are pulled. """
    self.host = self._get_host()
    """ Result of host calculations. """

    # must be last. Can't use porperty setter.
    self._epsilon = epsilon
    self._pa_kwargs = pa_kwargs


  @property 
  def epsilon(self): 
    """ Dimensionless dielectric constant. """
    if self._epsilon is not None: return self._epsilon
    if '/dielectric' in self.massextract and len(self.massextract['dielectric']) == 1:
      from numpy import trace
      epsilon = self.massextract.copy(naked_end=True)['dielectric'].dielectric_constant
      if epsilon is None: return 1
      if len(epsilon) != 3: return 1
      self.epsilon = trace(epsilon)/3e0
      return self._epsilon
    return 1e0

  @epsilon.setter
  def epsilon(self, value):
    self._epsilon = value 
    for v in self.itervalues(): v.epsilon = self._epsilon

  @property 
  def pa_kwargs(self): 
    """ Dimensionless dielectric constant. """
    return self._pa_kwargs
  @pa_kwargs.setter
  def pa_kwargs(self, value):
    self._pa_kwargs = value 
    for v in self.itervalues(): v.pa_kwargs = self._pa_kwargs


  @property
  def rootdir(self): 
    """ Path to the root-directory containing the poin-defects. """
    return self.massextract.rootdir
  @rootdir.setter
  def rootdir(self, value): self.massextract.rootdir = value
    
  def _get_host(self):
    """ Returns extraction object towards the host. """
    from operator import itemgetter
    host = self.massextract.copy(excludes=[".*PointDefects", ".*dielectric"], naked_end=False)
    host.excludes.extend(self.massextract.excludes)
    lowest = sorted(host.total_energies.iteritems(), key=itemgetter(1))[0][0]
    host = [u for u in host[lowest].itervalues()]
    assert len(host) == 1
    return host[0]


  def __iter_alljobs__(self):
    """ Walks through point-defects only. """
    for child in self.massextract["PointDefects"].children:
      # looks for site_n
      if len(child["site_\d+"].keys()) != 0:
        assert len(child["site_\d+"].keys()) == len(child.keys()),\
               RuntimeError("Don't understand directory structure of {0}.".format(child.view))
        for site in child.children: # should site specific defects.
          result = self.DefectExtractor( site, epsilon=self.epsilon,
                                         host=self.host, pa_kwargs=self.pa_kwargs )
          # checks this is a point-defect.
          if result.is_interstitial or result.is_vacancy or result.is_substitution:
            yield site.view, result
      else:
        result = self.DefectExtractor( child, epsilon=self.epsilon,
                                       host=self.host, pa_kwargs=self.pa_kwargs )
        # checks if this is a valid point-defect.
        if result.is_interstitial or result.is_vacancy or result.is_substitution:
          yield child.view, result

  def ordered_items(self):
    """ Returns items ordered by substitution, vacancy, and interstitial. """
    from operator import itemgetter
    interstitials = (u for u in self.iteritems() if u[1].is_interstitial)
    substitution  = (u for u in self.iteritems() if u[1].is_substitution)
    vacancy       = (u for u in self.iteritems() if u[1].is_vacancy)
    result = sorted(substitution, key = itemgetter(0)) 
    result.extend(sorted(vacancy, key = itemgetter(0)))
    result.extend(sorted(interstitials, key = itemgetter(0)))
    return result
  def ordered_keys(self):
    """ Returns keys ordered by substitution, vacancy, and interstitial. """
    return [u[0] for u in self.ordered_items()]
  def ordered_values(self):
    """ Returns values ordered by substitution, vacancy, and interstitial. """
    return [u[1] for u in self.ordered_items()]

        


class Material(_MaterialNavigator):
  """ Extracts data for a whole material. """
  def __init__(self, *args, **kwargs):
    """ Initializes an enthalpy function. """
    super(Material, self).__init__(*args, **kwargs)

  @property
  def cbm(self):
    """ Conduction band minimum of the host. """
    return self.host.cbm
  @property
  def vbm(self):
    """ Valence band maximum of the host. """
    return self.host.vbm

  def enthalpies(self, fermi, mu=None):
    """ Dictionary of point-defect formation enthalpies. 
    
        :Parameters:
          fermi  
            Fermi energy with respect to the host's VBM. If without
            units, assumes eV. 
          mu : dictionary or None
            Dictionary of chemical potentials. If without units, assumes eV.
            If None, chemical potential part of the formation enthalpy is
            assumed zero.

        :return: Dictionary where keys are the name of the defects, and the
          values the formation enthalpy.
    """
    from quantities import eV
    results = {}
    for name, defect in self.iteritems():
      results[name] = defect.enthalpy(fermi, mu).rescale(eV)
    return results
  
  def __str__(self): 
    """ Prints out all energies and corrections. """
    return "".join( str(value) for value in self.ordered_values() )
      
