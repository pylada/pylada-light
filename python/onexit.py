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

""" Holds a list of callbacks to call before exit. 

    Helps declares functions which are called if python exists abruptly.
    These functions are given a unique identifier so they can be removed if not
    needed anymore.
"""
from atexit import register

_callback_dict = {}
""" List of callbacks + arguments. """


def _call_callbacks():
  """ Calls on-exit functions. """
  try: 
    _callback_dict.pop('abort', None)
    _callback_dict.pop('term', None)
    while len(_callback_dict):
      name, (callback, args, kwargs) = _callback_dict.popitem()
      if callback is not None: 
        try: callback(*args, **kwargs)
        except: pass
  except: pass

@register
def _atexit_onexit(): 
  """ Specific at-exit function for pylada. """
  _call_callbacks()


def _onexit_signal(signum, stackframe):
  from signal import SIGABRT, SIGTERM, signal, SIG_DFL

  abort = _callback_dict.pop('abort', None)
  term  = _callback_dict.pop('term', None)

  _call_callbacks()

  if signum == SIGABRT and abort is not None:
    try: signal(SIGABRT, abort)
    except: signal(SIGABRT, SIG_DFL)
  elif signum == SIGTERM and term is not None: 
    try: signal(SIGTERM, term)
    except: signal(SIGTERM, SIG_DFL)
  else: signal(SIGABRT, SIG_DFL)
  raise SystemExit(signum)

# delete register from module
del register

def add_callback(callback, *args, **kwargs):
  """ Adds function to call at exit. 
  
      :param callback:
        Callback function.
      :param *args: 
        Arguments to the callback.
      :param **kwargs:
        Keyword arguments to the callback.

      :return: An identifier with which the callback can be deleted.
  """
  from uuid import uuid4
  id = uuid4()
  _callback_dict[id] = callback, args, kwargs
  return id

def del_callback(id):
  """ Deletes a callback from the list. """
  _callback_dict.pop(id, None)

# on first opening this module, change sigterm signal.
if len(_callback_dict) == 0:
  from signal import SIGABRT, SIGTERM, signal
  _callback_dict['abort'] = signal(SIGABRT, _onexit_signal)
  _callback_dict['term'] = signal(SIGTERM, _onexit_signal)
  del signal
  del SIGABRT
  del SIGTERM
