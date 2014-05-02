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

""" Magic function to copy a functional or subtree. """

__docformat__ = "restructuredtext en"
def copy_folder(self, event):
  """ Copies a jobfolder somewhere. 


      Emulates bash's ``cp`` command. By default, it only copies one job.
      However, it can be made to copy a full tree as well, using the ``-r``
      directive.  When it finds it would overwrite a non-empty jobfolder, it
      prompts the user interactively, unless the '-f' directive is given.


      >>> copyfolder thisjob thatjob

      Copies ``thisjob`` to ``thatjob``, but not the subfolders of ``thisjob``.

      By default, the jobfolders are deepcopied. This means that the none of
      the arguments or functionals of the current job are shared by the copied
      jobs. However, the relation-ships between the current jobs are retained
      in the destinations. In other words, if jobs 'JobA' and 'JobA/JobB' share
      the same 'structure' variable object and they are copied to 'JobC' (with
      '-r'), then two new subfolders are created, 'JobC/JobA'
      and'JobC/JobA/JobB'. Both of these subfolders will share a reference to
      the same `structure` object.  However their `structure` object is
      different from that of the original `JobA` and `JobB`. This feature can
      be turned off with the option `--nodeepcopy` (in which case, `structure`
      would be shared by all source and destination folders). Furthermore, if
      '--nodeepcopy' is used, then the functionals are shared between source
      and destination.
  """
  from argparse import ArgumentParser
  from os.path import join, normpath, relpath
  from copy import deepcopy
  from ..interactive import jobfolder as cjf

  parser = ArgumentParser(prog='%copyfolder',
               description='Copies a jobfolder from one location to another')
  parser.add_argument('-f', '--force', action='store_true', dest='force',
               help='Does not prompt when overwriting a job-folder.')
  parser.add_argument('-r', '--recursive', action='store_true',
               dest='recursive', help='Whether to copy subfolders as well.')
  parser.add_argument('--nodeepcopy', action='store_true',
               help='Destination folders will share the parameters '           \
                    'from the original folders.')
  parser.add_argument('source', type=str, metavar='SOURCE',
               help='Jobfolder to copy')
  parser.add_argument('destination', type=str, metavar='DESTINATION',
               help='Destination folder')

  try: args = parser.parse_args(event.split())
  except SystemExit: return None

  shell = get_ipython()

  # gets current folder.
  if 'jobparams' not in shell.user_ns:
    print 'No jobfolder currently loaded.'
    return 
  jobparams = shell.user_ns['jobparams']

  # normalize destination.
  if args.destination[0] != ['/']:
    destination = normpath(join(cjf.name, args.destination))
    if destination[0] != '/':
      print 'Incorrect destination', destination
      return 
  else: destination = normpath(args.destination)

  # create list of source directories
  if args.source[0] != ['/']:
    source = normpath(join(cjf.name, args.source))
    if source[0] != '/':
      print 'Incorrect source', source
      return 
  else: source = normpath(args.source)
  if source not in cjf: 
    print 'Source', source, 'does not exist'
    return
  if destination == source: 
    print 'Source and destination are the same'
    return 
  rootsource = source
  pairs = []
  if cjf[rootsource].is_executable and not cjf[rootsource].is_tagged:
    pairs = [(source, relpath(destination, rootsource))]
  if args.recursive:
    for source in jobparams[rootsource]:
      if not cjf[source].is_executable: continue
      if cjf[source].is_tagged: continue
      pairs.append((source, join(destination, relpath(source, rootsource))))
  if len(pairs) == 0:
    print "Nothing to copy."
    return 

  # now performs actual copy
  root = deepcopy(cjf.root) if not args.nodeepcopy else cjf.root
  for source, destination in pairs:
    # gets the jobfolder source.
    jobsource = root[source]
    # gets the jobfolder destination.
    jobdest = cjf
    for name in destination.split('/'): jobdest = jobdest / name
    # something already exists here
    if jobdest.is_executable and not args.force: 
      print 'Copying', jobsource.name, 'to', jobdest.name
      a = ''
      while a not in ['n', 'y']:
        a = raw_input( '{0} already exists. Overwrite? [y/n]'
                       .format(jobdest.name) )
      if a == 'n': 
        print jobdest.name, 'not overwritten.'
        continue
    # now copies folder items.
    for key, value in jobdest.__dict__.iteritems(): 
      if key not in ['children', 'parent', 'param']:
        jobdest.__dict__[key] = value
    jobdest._functional = jobsource._functional
    jobdest.params = jobsource.params.copy()

def delete_folder(self, event):
  """ Deletes a job-folder.
  
      By default, only the job itself is delete. If there are no sub-folders,
      then jobfolder itself is also deleted. Suppose we have a jobfolder
      '/JobA' and '/JobA/JobB' and both contain actual jobs. 

      >>> deletefolder /JobA

      This would remove the job-parameters from 'JobA' but leave '/JobA/JobB'
      unscathed. However, 

      >>> deletefolder /JobA/JobB

      will remove the branch 'JobB' completely, since there are no sub-folders
      there.

      It is also possible to remove all folders of a branch recursively:

      >>> deletefolder -r /JobA

      This will now remove JobA and all its subfolders.

      .. warning::
      
        This magic function does not check whether job-folders are 'on' or
        'off'. The recursive option should be used with care.
  """
  from argparse import ArgumentParser
  from os.path import join, normpath
  from ..interactive import jobfolder as cjf

  parser = ArgumentParser(prog='%deletefolder',
               description='Deletes a job-folder.')
  parser.add_argument('-f', '--force', action='store_true', dest='force',
               help='Does not prompt before deleting folders.')
  parser.add_argument('-r', '--recursive', action='store_true',
               dest='recursive', help='Whether to delete subfolders as well.')
  parser.add_argument('folder', type=str, metavar='JOBFOLDER',
               help='Jobfolder to delete')

  try: args = parser.parse_args(event.split())
  except SystemExit: return None

  shell = get_ipython()

  # normalize  folders to delete
  if args.folder[0] != ['/']:
    folder = normpath(join(cjf.name, args.folder))
  else: folder = normpath(args.folder)
  if folder not in cjf: 
    print "Folder", folder, "does not exist."
    return
  # deletes jobfolder recursively.
  if args.recursive:
    if not args.force:
      a = ''
      while a not in ['n', 'y']:
        a = raw_input( "Delete {0} and its subfolders? [y/n]"
                       .format(cjf[folder].name) )
      if a == 'n':
        print cjf[folder].name, "not deleted."
        return
    jobfolder = cjf[folder]
    if jobfolder.parent is None: 
      jobfolder = jobfolder.parent
      jobfolder._functional = None
      jobfolder.children = {}
      jobfolder.params = {}
    else: del jobfolder.parent[jobfolder.name.split('/')[-2]]
  # only delete specified jobfolder.
  else:
    if not args.force:
      a = ''
      while a not in ['n', 'y']:
        a = raw_input("Delete {0}? [y/n]".format(cjf[folder].name))
      if a == 'n':
        print cjf[folder].name, "not deleted."
        return
    jobfolder = cjf[folder]
    if len(jobfolder.children) == 0 and jobfolder.parent is not None:
      del jobfolder.parent[jobfolder.name.split('/')[-2]]
    else:
      jobfolder._functional = None
      jobfolder.params = {}


def copy_completer(self, event):
  from IPython import TryNext
  from pylada import interactive

  line = event.line.split()[1:]
  result = []
  if '-h' not in line and '--help' not in line: result.append('-h')
  if '-r' not in line and '--recursive' not in line: result.append('-r')

  noop = [k for k in line if k[0] != '-']
  if len(noop) > 2: raise TryNext()

  if '/' in event.symbol:
    subkey = ""
    for key in event.symbol.split('/')[:-1]: subkey += key + "/"
    try: subdict = interactive.jobfolder[subkey]
    except KeyError: raise TryNext
    if hasattr(subdict, "children"): 
      if hasattr(subdict.children, "keys"):
        return [subkey + a + "/" for a in subdict.children.keys()]
    raise TryNext()
  else:
    result += [a + "/" for a in interactive.jobfolder.children.keys()]
    result.extend(["/"])
    if interactive.jobfolder.parent is not None: result.append("../")
    if len(getattr(interactive, "_pylada_subjob_iterated", [])) != 0:
      result.append("previous")
    return result

def delete_completer(self, event):
  from IPython import TryNext
  from pylada import interactive

  line = event.line.split()[1:]
  result = []
  if '-h' not in line and '--help' not in line: result.append('-h')
  if '-r' not in line and '--recursive' not in line: result.append('-r')

  noop = [k for k in line if k[0] != '-']
  if len(noop) > 1: return result
  elif len(noop) == 1 and len(event.symbol) == 0: return result

  if '/' in event.symbol:
    subkey = ""
    for key in event.symbol.split('/')[:-1]: subkey += key + "/"
    try: subdict = interactive.jobfolder[subkey]
    except KeyError: raise TryNext
    if hasattr(subdict, "children"): 
      if hasattr(subdict.children, "keys"):
        return [subkey + a + "/" for a in subdict.children.keys()]
    raise TryNext()
  else:
    result += [a + "/" for a in interactive.jobfolder.children.keys()]
    result.extend(["/"])
    if interactive.jobfolder.parent is not None: result.append("../")
    if len(getattr(interactive, "_pylada_subjob_iterated", [])) != 0:
      result.append("previous")
    return result
