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

""" Database related parameters. """
if "pyladabase" in globals()["pyladamodules"]:
  OUTCARS_prefix     = 'OUTCARs'
  """ Name of the collection of OUTCAR files. """
  vasp_database_name = 'cid'
  """ Name of the database. """
  username  = "Mayeul d'Avezac"
  """ Name with which to tag file in database. """
  # pymongo_username = 'mdadcast'
  # """ Username in the pymongo database. """
  pymongo_host = 'sccdev'
  """ Host of the database. """
  pymongo_port = 27016
  """ Port to which to connect on host. """
  local_push_dir = "/tmp/database_tmp"
  """ Directory where files are pushed, before being pulled to redrock. """
  pyladabase_doconnect = False
  """ Whether to connect to database when starting ipython. """
  add_push_magic_function = False
  """ Whether to the %push magic function to the IPython interface. 
  
      Thanks to the paranoids with insecurity issues, it may not be possible to
      access the database from just anywhere. Hence %push is machine dependent.  
  """
