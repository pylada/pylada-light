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

def savefolders(self, event):
  """ Saves job-folder to disk.
  
      This function can be called in one of three ways:

      >>> savefolders filename.dict rootfolder 
  
      In this case, "filename.dict" is a file where to save the jobfolder
      "rootfolder". The latter must be a python variable, not another filename.
      The current job-folder becomes "rootfolder", and the current path 


      >>> savefolders filename.dict

      Saves the current job-folder to "filename.dict". Fails if no current
      job-folder.

      >>> savefolders 

      Saves the current job-folder to the current job-folder path. Fails if
      either are unknown.
  """
  from os.path import exists, isfile
  from ..jobfolder import JobParams, MassExtract as Collect, save
  from .. import interactive
  from ..misc import RelativePath
  from . import get_shell

  shell = get_shell(self)

  args = [u for u in event.split()]
  if '--help' in args or '-h' in args:
    print savefolders.__doc__
    return 

  if len(args) > 2: 
    print "savefolders takes zero, one, or two arguments."
    return

  if len(args) == 2:
    from .explore import explore
    explore(self, args[1])
    savefolders(self, args[0])
    return

  if interactive.jobfolder is None: 
    print "No current job-folder."
    print "Please load first with %explore."
    return
  jobfolder = interactive.jobfolder.root
  jobfolder_path = interactive.jobfolder_path

  if len(args) == 1:
    jobfolder_path = RelativePath(args[0]).path
    interactive.jobfolder_path = jobfolder_path

  if jobfolder_path is None: 
    print "No current job-folder path.\n"\
          "Please specify on input, eg\n"\
          ">saveto this/path/filename"
    return
  if exists(jobfolder_path): 
    if not isfile(jobfolder_path): 
      print "{0} is not a file.".format(jobfolder_path)
      return
    a = 'y'       # testValidProgram: force yes to allow automated testing
    while a not in ['n', 'y']:
      a = raw_input( "File {0} already exists.\nOverwrite? [y/n] "             \
                     .format(jobfolder_path) )
    if a == 'n':
      print "Aborting."
      return
  save(jobfolder.root, jobfolder_path, overwrite=True, timeout=10) 
  if len(args) == 1:
    if "collect" not in shell.user_ns:
      shell.user_ns["collect"] = Collect(dynamic=True, path=jobfolder_path)
    if "jobparams" not in shell.user_ns:
      shell.user_ns["jobparams"] = JobParams()
