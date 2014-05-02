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

""" Launches jobs as a single job-array. """
__docformat__ = 'restructuredtext en'

def launch(self, event, jobfolders):
  """ Launches each job-folder as an array pbs script. """
  for jobfolder in jobfolders: launch_single(self, event, jobfolder)

def launch_single(self, event, jobfolder):
  """ Launches a single jobfolder as a pbs array job """
  import subprocess
  from copy import deepcopy
  from os.path import dirname, join, basename, exists
  from os import remove
  from ... import pbs_string, default_pbs, qsub_array_exe, default_comm
  from . import get_walltime, get_queues, get_mppalloc
  from . import array_script

  shell = get_ipython()

  pbsargs = deepcopy(dict(default_comm))
  pbsargs.update(default_pbs)
  pbsargs['ppn'] = event.ppn
  if not get_walltime(shell, event, pbsargs): return
  if not get_queues(shell, event, pbsargs): return

  # gets python script to launch in pbs.
  pyscript = array_script.__file__
  if pyscript[-1] == 'c': pyscript = pyscript[:-1]

  # creates file names.
  hasprefix = getattr(event, "prefix", None)                               
  def pbspaths(directory, jobname, suffix):
    """ creates filename paths. """
    name = '{0}-{1}{2}'.format(event.prefix, jobname, suffix) if hasprefix     \
           else '{0}{1}'.format(jobname, suffix)
    return join(directory, name)

  # now  loop over jobfolders
  current, path  = jobfolder
  # Check number of jobs
  directory, nbjobs = dirname(path), 0
  lines = []
  for name, job in current.root.iteritems():
    # avoid jobfolder which are off
    if job.is_tagged: continue
    # avoid successful jobs.unless specifically requested
    if hasattr(job.functional, 'Extract') and not event.force: 
      p = join(directory, name)
      extract = job.functional.Extract(p)
      if extract.success:
        print "Job {0} completed successfully. "                               \
              "It will not be relaunched.".format(name)                     
        continue                                                            
    lines.append(name)
    nbjobs += 1
  if path.rfind('.') != -1: name = path[:path.rfind('.')]
  # now creates script
  pbsargs['n'] = get_mppalloc(shell, event, False)                                              
  pbsargs['nnodes'] = (pbsargs['n'] + pbsargs['ppn'] - 1)                      \
                      // pbsargs['ppn']                                     
  pbsargs['err'] = pbspaths(directory, name, '.err')                    
  pbsargs['out'] = pbspaths(directory, name, '.out')
  pbsargs['name'] = basename(path)
  pbsargs['directory'] = directory                                     
  pbsargs['header'] = ""
  for i, line in enumerate(lines):
    pbsargs['header'] += 'JOB[{0}]={1!r}\n'.format(i+1, line)
  pbsargs['header'] += '\nexport PYLADA_JOBARRAY_NAME=${{JOB[{0}]}}\n'           \
                       .format(qsub_array_exe[1])
  pbsargs['scriptcommand'] = "{0} --nbprocs {n} --ppn {ppn} {1} "              \
                             .format(pyscript, path, **pbsargs)                      
  pbsscript = pbspaths(directory, name, '.script')
                                                                       
  # write pbs scripts                                                  
  if exists(pbsscript):
    a = ''
    while a not in ['n', 'y']:
      a = raw_input( "PBS script {0} already exists.\n"                        \
                     "Only one array job per jobfolder can be launched at"     \
                     "a time.\nAre you sure this job is not currently running" \
                     "[y/n]? ".format(pbsscript) )
    if a == 'n':
      print "Aborting."
      return
    remove(pbsscript)                    
  with open(pbsscript, "w") as file:                              
    string = pbs_string(**pbsargs) if hasattr(pbs_string, '__call__')          \
             else pbs_string.format(**pbsargs)                         
    file.write(string)                                                 
  assert exists(pbsscript)                                        
  print "Created pbsscript {0} for job-folder {1}."                            \
        .format(pbsscript, path)

  if event.nolaunch: return
  # otherwise, launch.
  cmdLine = "{0} {1}".format(
    qsub_array_exe[0].format(nbjobs=nbjobs), pbsscript)
  subprocess.call( cmdLine, shell=True)

def completer(self, info, data):
  """ Completer for scattered launcher. """
  from .. import jobfolder_file_completer
  from ... import queues, accounts, debug_queue
  if len(data) > 0: 
    if data[-1] == "--walltime":
      return [ u for u in self.user_ns                                         \
               if u[0] != '_' and isinstance(self.user_ns[u], str) ]
    elif data[-1] == "--nbprocs": return ['']
    elif data[-1] == '--ppn':     return ['']
    elif data[-1] == "--prefix":  return ['']
    elif data[-1] == "--queue":   return queues
    elif data[-1] == "--account": return accounts
  result = ['--force', '--walltime', '--nbprocs', '--help', '--ppn']
  if len(queues) > 0: result.append("--queue") 
  if len(accounts) > 0: result.append("--account") 
  if debug_queue is not None: result.append("--debug")
  result.extend(jobfolder_file_completer([info.symbol]))
  result = list(set(result) - set(data))
  return result

def parser(self, subparsers, opalls):
  """ Adds subparser for scattered. """ 
  from ... import default_comm 
  from . import set_queue_parser, set_default_parser_options
  result = subparsers.add_parser( 'array', 
              description="Calculations are launched in a single job array.",
              parents=[opalls])
  set_default_parser_options(result)
  result.add_argument( '--nbprocs', type=str, dest="nbprocs", required=True,
              help="Total number of processors per pbsjob/jobfolder" )
  result.add_argument( '--ppn', dest="ppn",
              default=default_comm.get('ppn', 1), type=int,
              help="Number of processes per node. Defaults to {0}."            \
                   .format(default_comm.get('ppn', 1)))
  set_queue_parser(result)
  result.set_defaults(func=launch)
  return result
