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

""" Module to decorate properties with json transcripters. """

def section(name):
  """ Adds name of section where this property should be located in json. 
  
      None means data should be added to head dictionary.
  """
  def add_section_name(function): 
    function.section = name
    return function
  return add_section_name 

def unit(unit):
  """ Creates JSON transfer functions wich remove/add units. """
  def to_json(object):
    """ Removes unit from object. """
    return object.rescale(unit).magnitude.tolist()
  def from_json(object):
    """ Adds unit to object. """
    return object * unit
  def add_json_transfer_functions(function):
     """ Adds json transfer functions to an object. """
     function.to_json = to_json
     function.from_json = from_json
     return function
  return add_json_transfer_functions

def array(type):
  """ Creates JSON transfer functions wich transforms numpy arrays to list. """
  def to_json(object):
    """ Transforms array to list. """
    return object.tolist()
  def from_json(object):
    """ Transforms list to array. """
    from numpy import array
    return array(object, dtype=type)
  def add_json_transfer_functions(function):
     """ Adds json transfer functions to an object. """
     function.to_json = to_json
     function.from_json = from_json
     return function
  return add_json_transfer_functions

def array_with_unit(type, unit):
  """ Creates JSON transfer functions wich transforms numpy arrays to list. """
  def to_json(object):
    """ Transforms array to list. """
    return object.magnitude.tolist()
  def from_json(object):
    """ Transforms list to array. """
    from numpy import array
    return array(object, dtype=type) * unit
  def add_json_transfer_functions(function):
     """ Adds json transfer functions to an object. """
     function.to_json = to_json
     function.from_json = from_json
     return function
  return add_json_transfer_functions


def pickled(function):
  """ Adds JSON transfer functions which work through a pickle. """
  from pickle import dumps, loads
  def to_json(object):
    """ Transform to pickle. """
    return dumps(object)
  def from_json(object):
    """ Transform to pickle. """
    return loads(str(object))
  function.to_json = to_json
  function.from_json = from_json
  return function
