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

def listfolders(self, arg):
  """ Lists sub-folders. """
  from fnmatch import fnmatch
  from pylada import interactive
  from ..jobfolder import JobParams
  if interactive.jobfolder is None: return
  if len(arg) == 0:
    string = ''
    for i, name in enumerate(interactive.jobfolder.children.iterkeys()):
      string += name + '  '
      if (i+1) % 6 == 0: string += '\n'
    print string if len(string) != 0 else "No sub-folders."
    return
  elif 'all' in arg.split():
    current = JobParams(jobfolder=interactive.jobfolder)[interactive.jobfolder.name]
    for job in current.jobfolder.root.itervalues():
      if job.is_tagged: continue
      print job.name
    return
  else:
    dirs = arg.split('/')
    result = set()
    for name in interactive.jobfolder.iterleaves():
      name = name[len(interactive.jobfolder.name):]
      if len(name) == 0: continue
      names = name.split('/')
      if len(names) < len(dirs): continue
      if all(fnmatch(u,v) for u,v in zip(names, dirs)): 
        result.add('/'.join(names[:len(dirs)]))
    for i, string in enumerate(result):
      print string,
      if (i+1) % 6 == 0: print '\n'
