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

""" Module providing an interface to VASP code.

    The interface is separated into 4 conceptual areas:
      - VASP parameterization: mostly contained within :py:mod:`incar <pylada.vasp.incar>` submodule
      - Launching vasp: a single-shot run is performed with a :py:class:`Vasp` object
      - Extracting data from vasp output: to be found in :py:mod:`extract <pylada.vasp.extract>` submodule
      - Methods: One can chain vasp runs together for more complex calculations

    The :py:class:`Vasp <pylada.vasp.functional.Vasp>` class  combines the first
    three concepts together.  It allows us to launch vasp and retrieve
    information from the output. It checks for errors and avoids running the
    same job twice. Hence data retrieval and vasp calculations can be performed
    using the same class and script. 

    `version` tells for which version of VASP these bindings have been
    compiled.
"""
__docformat__ = "restructuredtext en"
__all__ = ['Vasp', 'Extract', 'Specie', 'MassExtract', 'relax', 'emass', 'read_input', 'exec_input']
from .extract import Extract, MassExtract
from .specie import Specie
from .functional import Vasp
from . import relax, emass

def read_input(filepath="input.py", namespace=None):
  """ Specialized read_input function for vasp. 
  
      :Parameters: 
        filepath : str
          A path to the input file.
        namespace : dict
          Additional names to include in the local namespace when evaluating
          the input file.

      It add a few names to the input-file's namespace. 
  """
  from ..misc import read_input
  from . import specie
  from relax import Epitaxial, Relax

  # names we need to create input.
  input_dict = {"Vasp": Vasp, "U": specie.U, "nlep": specie.nlep, 'Extract': Extract, 
                'Relax': Relax, 'Epitaxial': Epitaxial }
  if namespace is not None: input_dict.update(namespace)
  return read_input(filepath, input_dict)

def exec_input( script, global_dict=None, local_dict=None,
                paths=None, name=None ):
  """ Specialized exec_input function for vasp. """
  from ..misc import exec_input

  # names we need to create input.
  if global_dict is None: global_dict = {}
  for k in __all__:
    if k != 'read_input' and k != 'exec_input': global_dict[k] = globals()[k]
  return exec_input(script, global_dict, local_dict, paths, name)

def parse_incar(path):
  """ Reads INCAR file and returns mapping (keyword, value). """
  from os.path import isdir, join
  from ..error import ValueError
  from ..misc import RelativePath
  from ..tools.input import Tree
  if isinstance(path, str): 
    if path.find('\n') == -1:
      dummy = RelativePath(path).path
      if isdir(dummy): dummy = join(dummy, 'INCAR')
      with open(dummy) as file: return parse_incar(file)
    else:
      return parse_incar(path.split('\n').__iter__())
  
  lines = []
  for line in path:
    if line.find('#') != -1: line = line[:line.find('#')]
    dummy = [u.lstrip().rstrip() for u in line.split(';')]
    dummy = [u for u in dummy if len(u) > 0]
    if len(dummy) and len(lines) == 0: lines.append(dummy.pop(-1))
    while len(dummy):
      if lines[-1][-1] == '\\':
        lines[-1] = lines[-1][:-1] + dummy.pop(-1)
      else: lines.append(dummy.pop(-1))

  result = Tree()
  for line in lines:
    if line.find('=') == -1: continue
    keyword, value = [u.rstrip().lstrip() for u in line.split('=')]
    if len(keyword) == 0: raise ValueError('Found empty keword in INCAR.')
    if keyword in result: raise ValueError('Found duplicate keyword {0} in INCAR'.format(keyword))
    result[keyword] = value
  return result


def read_incar(filename='INCAR'):
  """ Reads a functional from an INCAR.

      :param filename:
        It can be one of the following:

          - An iterable object: each iteration should yield a line in the
            INCAR.
          - A string (single line): path to a directory containing an INCAR_ or
            to an INCAR_ file itself.
          - A string (multi-line): should be the content of an INCAR_.
        
        Defaults to 'INCAR'.
      :returns: A vasp functional equivalent to the INCAR_.
      :rtype: :py:class:`~pylada.vasp.functional.Vasp`
  """
  from .functional import Vasp
  parsed = parse_incar(filename)
  result = Vasp()
  result.read_input(parsed)
  return result
