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

""" Methods to compute effective masses and other derivatives. """
__docformat__ = "restructuredtext en"
__all__ = ["Extract", "iter_emass", 'EMass', 'effective_mass']
from .extract import Extract as ExtractDFT
from ..tools.makeclass import makeclass, makefunc
from ..tools import make_cached
from .functional import Vasp

class Extract(ExtractDFT):
  """ Extractor for reciprocal taylor expansion. """
  def __init__(self, *args, **kwargs):
     super(Extract, self).__init__(*args, **kwargs)

  @property 
  def success(self):
    """ True if successful run. 
    
        Checks this is an effective mass calculation.
    """
    from .extract import Extract as ExtractDFT
    try: self._details
    except: return False
    return ExtractDFT.success.__get__(self)

  @property
  @make_cached
  def _details(self):
    """ Parameters when calling the effective mass routine. """
    from re import compile
    from ..misc import exec_input
    from ..error import GrepError
    start = compile(r'^#+ EMASS DETAILS #+$')
    end = compile(r'^#+ END EMASS DETAILS #')
    with self.__outcar__() as file:
      lines = None
      for line in file:
        if start.match(line): lines = ""; break
      if lines is None: raise GrepError('Could not find call parameters.')
      for line in file:
        if end.match(line):  break
        lines += line

    input = exec_input(lines)
    return { 'center': input.center,
             'nbpoints': input.nbpoints,
             'input': input.input,
             'range': input.range }
  @property
  def center(self): return self._details['center']
  @property
  def nbpoints(self): return self._details['nbpoints']
  @property
  def range(self): return self._details['range']
  @property
  def input(self): return self._details['input']

  @staticmethod
  def _orders(orders):
    """ Order up to which taylor coefficients should be computed. """
    result = orders 
    if result is None: result = [0, 2]
    if not hasattr(result, '__iter__'): result = [result]
    return sorted(result)

  @property
  def breakpoints(self):
    """ Indices for start of each path. """
    from numpy import any, abs, cross
    breakpoints, last_dir = [0], None
    for i, k in enumerate(self.kpoints[1:]):
      if last_dir is None: last_dir = k - self.kpoints[breakpoints[-1]]
      elif any( abs(cross(last_dir, k-self.kpoints[breakpoints[-1]])) > 1e-8):
        breakpoints.append(i+1)
        last_dir = None
    return breakpoints + [len(self.kpoints)]

  @property
  def directions(self):
    """ Direction for each path. """
    from numpy import array
    from numpy.linalg import norm
    from quantities import angstrom
    results = []
    breakpoints = self.breakpoints
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
      results.append(self.kpoints[end-1] - self.kpoints[start])
      results[-1] /= norm(results[-1])
    return array(results) / angstrom

  def emass(self, orders=None):
    """ Computes effective mass for each direction. """
    from numpy import dot, concatenate, pi, array
    from numpy.linalg import inv, lstsq
    from math import factorial
    from quantities import angstrom, emass, h_bar
    from ..error import ValueError

    orders = self._orders(orders)
    if 2 not in orders:
      raise ValueError('Cannot compute effective masses without second order term.')

    results = []
    breakpoints = self.breakpoints
    recipcell = inv(self.structure.cell).T * 2e0 * pi / self.structure.scale
    for start, end, direction in zip( breakpoints[:-1], 
                                      breakpoints[1:], 
                                      self.directions ):
      kpoints = self.kpoints[start:end]
      x = dot(direction, dot(recipcell, kpoints.T)) 
      measurements = self.eigenvalues[start:end].copy()
      parameters = concatenate([x[:, None]**i / factorial(i) for i in orders], axis=1)
      fit = lstsq(parameters, measurements)
      results.append(fit[0][orders.index(2)])

    result = (array(results) * self.eigenvalues.units * angstrom**2 / h_bar**2)
    return 1./result.rescale(1/emass)

  def fit_directions(self, orders=None):
    """ Returns fit for computed directions.
    
        When dealing with degenerate states, it is better to look at each
        computed direction separately, since the order of bands might depend on
        the direction (in which case it is difficult to construct a tensor).
    """
    from numpy import dot, concatenate, pi
    from numpy.linalg import inv, lstsq
    from math import factorial
    from ..error import ValueError

    orders = self._orders(orders)
    if 2 not in orders:
      raise ValueError('Cannot compute effective masses without second order term.')

    results = []
    breakpoints = self.breakpoints
    recipcell = inv(self.structure.cell).T * 2e0 * pi / self.structure.scale
    for start, end, direction in zip( breakpoints[:-1], 
                                      breakpoints[1:], 
                                      self.directions ):
      kpoints = self.kpoints[start:end]
      x = dot(direction, dot(recipcell, kpoints.T)) 
      measurements = self.eigenvalues[start:end].copy()
      parameters = concatenate([x[:, None]**i / factorial(i) for i in orders], axis=1)
      fit = lstsq(parameters, measurements)
      results.append(fit[0])

    return results

  def iterfiles(self, **kwargs):
    """ Exports files from both calculations. """
    from itertools import chain
    for file in chain( super(Extract, self).iterfiles(**kwargs),
                       self.input.iterfiles(**kwargs) ):
      yield file

class _OnFinish(object):
  """ Called when effective mass calculation finishes. 
  
      Adds some data to the calculation so we can figure out what the arguments
      to the call.
  """
  def __init__(self, previous, outcar, details):
    super(_OnFinish, self).__init__()
    self.details  = details
    self.outcar   = outcar
    self.previous = previous
  def __call__(self, *args, **kwargs):
    # first calls previous onfinish.
    if self.previous is not None: self.previous(*args, **kwargs)
    # then adds data
    header = ''.join(['#']*20)
    with open(self.outcar, 'a') as file: 
      file.write('{0} {1} {0}\n'.format(header, 'EMASS DETAILS'))
      file.write('{0}\n'.format(self.details))
      file.write('{0} END {1} {0}\n'.format(header, 'EMASS DETAILS'))
    
def iter_emass( functional, structure=None, outdir=None, center=None,
                nbpoints=3, directions=None, range=0.1, emassparams=None,
                **kwargs ):
  """ Computes k-space taylor expansion of the eigenvalues up to given order.

      First runs a vasp calculation using the first input argument, regardless
      of whether a restart keyword argument is also passed. In practice,
      following the general Pylada philosophy of never overwritting previous
      calculations, this will not rerun a calculation if one exists in
      ``outdir``. 
      Second, a static non-self-consistent calculation is performed to compute
      the eigenvalues for all relevant kpoints.

      :param functional:
         Two types are accepted: 

         - :py:class:`~vasp.Vasp` or derived functional: a self-consistent  run
           is performed and the resulting density is used as to define the
           hamiltonian for which the effective mass is computed.
         - :py:class:`~vasp.Extract` or derived functional: points to the
           self-consistent calculations defining the hamiltonian for which the
           effective mass is computed.
      
      :param structure: The structure for wich to compute effective masses.
      :type structure: `~pylada.crystal._cppwrapper.Structure`

      :param center:
          Central k-point of the taylor expansion. This should be given in
          **reciprocal** units (eg coefficients to the reciprocal lattice
          vectors). Default is None and means |Gamma|.
      :type center: 3 floats

      :param str outdir:
          Root directory where to save results of calculation. Calculations
          will be stored in  "reciprocal" subdirectory of this input parameter.

      :param int nbpoints:
          Number of points (in a single direction) with wich to compute
          taylor expansion.  Should be at least order + 1. Default to 3.
      
      :param directions:
          Array of directions (cartesian coordinates). If None, defaults to a
          reasonable set of directions: 001, 110, 111 and so forth. Note that
          if given on input, then the tensors should not be extracted. The
          directions are normalized. Eventually, the paths will extend from
          ``directions/norm(directions)*range`` to
          ``-directions/norm(directions)*range``.
      :type directions: list of 3d-vectors or None

      :param float range:
          Extent of the grid around the central k-point.

      :param dict emassparams: 
         Parameters for the (non-self-consistent) effective mass caclulation
         proper. For instance, could include pertubative spin-orbit
         (:py:attr:`~vasp.functional.Vasp.lsorbit`).

      :param kwargs:
          Extra parameters which are passed on to vasp, both for the initial
          calculation and the effective mass calculation proper.

      :return: Extraction object from which masses can be obtained.

      .. |pi|     unicode:: U+003C0 .. GREEK SMALL LETTER PI
      .. |Gamma|  unicode:: U+00393 .. GREEK CAPITAL LETTER GAMMA
  """
  from copy import deepcopy
  from os import getcwd
  from os.path import join, samefile, exists
  from numpy import array, dot, arange, sqrt
  from numpy.linalg import inv, norm
  from ..error import input as InputError
  from ..misc import RelativePath
  from . import Vasp

  # save input details for printing later on.
  details = 'directions = {0!r}\n'                                             \
            'range      = {1!r}\n'                                             \
            'center     = {2!r}\n'                                             \
            'nbpoints   = {3!r}\n'                                             \
            .format(directions, range, center, nbpoints)
 
  # takes care of default parameters.
  if center is None: center = kwargs.pop("kpoint", [0,0,0])
  center = array(center, dtype="float64")
  if outdir is None: outdir = getcwd()

  # If has an 'iter' function, then calls it. 
  if hasattr(functional, 'iter'): 
    if structure is None:
      raise InputError( 'If the first argument to iter_emass is a functional, '\
                        'then a structure must also be given on which to '     \
                        'apply the CRYSTAL functional.' )
    for input in functional.iter(structure, outdir=outdir, **kwargs):
      if getattr(input, 'success', False): continue
      elif hasattr(input, 'success'):
        yield Extract(outdir)
        return
      yield input 
  # if is callable, then calls it.
  elif hasattr(functional, '__call__'):
    input = functional(structure, outdir=outdir, **kwargs)
  # otherwise, assume it is an extraction object.
  else: input = functional
  # creates a new VASP functional from the input.
  functional = Vasp(copy=input.functional)

  # check that self-consistent run was successful.
  if not input.success:
    yield input
    return

  # prepare second run.
  center = dot(inv(input.structure.cell).T, center)
  if directions is None:
    kpoints = array([ [1, 0, 0], [-1, 0, 0],
                    [0, 1, 0], [0, -1, 0],
                    [0, 0, 1], [0, 0, -1],
                    [1, 0, 1], [-1, 0, -1],
                    [0, 1, 1], [0, -1, -1],
                    [1, 1, 0], [-1, -1, 0],
                    [1, 0, -1], [-1, 0, 1],
                    [0, -1, 1], [0, 1, -1],
                    [-1, 1, 0], [1, -1, 0],
                    [1, 1, 1], [-1, -1, -1],
                    [1, 1, -1], [-1, -1, 1],
                    [1, -1, 1], [-1, 1, -1],
                    [-1, 1, 1], [1, -1, -1] ], dtype='float64')
    kpoints[6:18] *= 1e0/sqrt(2.)
    kpoints[18:] *= 1e0/sqrt(3.)
  else: 
    directions = array(directions).reshape(-1, 3)
    directions = array([array(d)/norm(d) for d in directions])
    points = arange(-0.5, 0.5 + 1e-8, 1.0/float(nbpoints))
    kpoints = array([d * p for d in directions for p in points])

  functional.kpoints = kpoints * range + center
  
  # and exectute it.
  # onfinish is modified so that parameters are always included.
  kwargs = deepcopy(kwargs)
  kwargs['restart'] = input
  kwargs['nonscf']  = True
  kwargs['relaxation']  = None
  if emassparams is not None: kwargs.update(emassparams)
  if outdir is None: outdir = getcwd()
  if outdir is not None:
    if exists(outdir) and samefile(outdir, input.directory): 
      outdir = join(input.directory, "reciprocal")
  # saves input calculations into the details
  if isinstance(getattr(input, '_directory', None), RelativePath):
    input = deepcopy(input)
    input._directory.envvar = outdir
    details += 'from {0.__class__.__module__} import {0.__class__.__name__}\n' \
               .format(input)
    details += 'input = {0!r}\n'.format(input)
  for u in functional.iter(input.structure, outdir=outdir, **kwargs):
    if getattr(u, 'success', False): continue
    if hasattr(u, 'success'): yield u; return
    # modify onfinish so that call arguments are added to the output file.
    onfinish = _OnFinish(u.onfinish, join(outdir, 'OUTCAR'), details) 
    u.onfinish = onfinish
    yield u

  yield iter_emass.Extract(outdir)

iter_emass.Extract = Extract
""" Extractor class for the reciprocal method. """
EMass = makeclass( 'EMass', Vasp, iter_emass, None, module='pylada.vasp.emass',
                    doc='Functional form of the '                              \
                        ':py:class:`pylada.emass.relax.iter_emass` method.' )

# Function call to effective mass. No iterations. returns when calculations are
# done or fail.
effective_mass = makefunc('effective_mass', iter_emass, 'pylada.vasp.emass')
effective_mass.Extract = iter_emass.Extract

del makefunc
del makeclass
del ExtractDFT
del make_cached
del Vasp
