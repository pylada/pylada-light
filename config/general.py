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

""" Sets general pylada parameters. """
jobparams_readonly = False
""" Whether items can be modified in parallel using attribute syntax. """
jobparams_naked_end = True
""" Whether last item is returned as is or wrapped in ForwardingDict. """
jobparams_only_existing = True
""" Whether attributes can be added or only modified. """
unix_re  = True
""" If True, then all regex matching is done using unix-command-line patterns. """
verbose_representation = True
""" Whether functional should be printed verbosely or not. """
ipython_verbose_representation = False
""" When in ipython, should we set :py:data:`verbose_representation` to False. """

global_root = '/'
""" Root of relative paths. 

    This can be set an environment variable, say "$PYLADA" to make it easier to
    transfer job-dictionaries from one computer to another. All file paths in
    Pylada are then given with respect to this one. As long as the structure of
    the disk is the same relative to this path, all Pylada paths will point to
    equivalent objects.
"""
global_tmpdir = None
""" Global temporary directory for Pylada.

    If None, defaults to system tmp dir. However, two environment variable take
    precedence: PBS_TMPDIR and PYLADA_TMPDIR.
"""


