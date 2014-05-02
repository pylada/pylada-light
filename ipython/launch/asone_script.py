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

""" Run jobfolder using JobFolderProcess """
def main():
  import re 
  from sys import path as python_path
  from os import getcwd
  from os.path import exists
  from argparse import ArgumentParser
  from pylada import jobfolder
  from pylada.process.mpi import create_global_comm
  from pylada.process.jobfolder import JobFolderProcess
  import pylada

  # below would go additional imports.

  parser = ArgumentParser( prog="runasone", description = re.sub("\\s+", " ", __doc__[1:]))
  parser.add_argument( "--pools", type=int, default=0, help="Number of pools" )
  parser.add_argument( "--ppath", dest="ppath", default=None, \
                       help="Directory to add to python path",
                       metavar="Directory" )
  parser.add_argument('--nbprocs', dest="nbprocs", default=pylada.default_comm['n'], type=int,\
                      help="Number of processors with which to launch job.")
  parser.add_argument('--ppn', dest="ppn", default=pylada.default_comm['ppn'], type=int,\
                      help="Number of processors with which to launch job.")
  parser.add_argument('--timeout', dest="timeout", default=300, type=int,\
                      help="Time to wait for job-dictionary to becom available "
                           "before timing out (in seconds). A negative or null "
                           "value implies forever. Defaults to 5mn.")
  parser.add_argument('pickle', metavar='FILE', type=str, help='Path to a job-folder.')

  try: options = parser.parse_args()
  except SystemExit: return

  # additional path to look into.
  if options.ppath is not None: python_path.append(options.ppath)

  if not exists(options.pickle): 
    print "Could not find file {0}.".format(options.pickle)
    return

  # Set up mpi processes.
  pylada.default_comm['ppn'] = options.ppn
  pylada.default_comm['n'] = options.nbprocs
  create_global_comm(options.nbprocs)

  timeout = None if options.timeout <= 0 else options.timeout
  
  jobfolder = jobfolder.load(options.pickle, timeout=timeout)
  process = JobFolderProcess(jobfolder, outdir=getcwd(), nbpools=options.pools)
  process.start(pylada.default_comm)
  process.wait(60)

if __name__ == "__main__": main()
