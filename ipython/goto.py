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

""" IPython goto and iterate magic functions. """
def goto(self, cmdl):
  """ Moves current dictionary position and working directory (if appropriate). """
  from os import chdir
  from os.path import exists, join, split as splitpath, isdir
  from .explore import explore
  from . import get_shell
  from pylada import interactive

  shell = get_shell(self)

  if interactive.jobfolder is None: 
    print "No current job-folders."
    return
  if len(cmdl.split()) == 0:
    if interactive.jobfolder_path is None:
      print "Current position in job folder:", interactive.jobfolder.name
    else:
      print "Current position in job folder:", interactive.jobfolder.name
      print "Filename of job-folder: ", interactive.jobfolder_path
    return
  args = cmdl.split()
  if len(args) > 1:
    print "Invalid argument to goto {0}.".format(cmdl)
    return

  # if no argument, then print current job data.
  if len(args) == 0: 
    explore(self, "")
    return

  # cases to send to iterate.
  if args[0] == "next":       return iterate(self, "")
  elif args[0] == "previous": return iterate(self, "previous")
  elif args[0] == "reset":    return iterate(self, "reset")

  # case for which precise location is given.
  try: result = interactive.jobfolder[args[0]] 
  except KeyError as e: 
    print e
    return 

  interactive.jobfolder = result
  if 'jobparams' in shell.user_ns:
    shell.user_ns['jobparams'].view = interactive.jobfolder.name
  if 'collect' in shell.user_ns:
    shell.user_ns['collect'].view = interactive.jobfolder.name
  good = not interactive.jobfolder.is_tagged
  if good:
    for value in interactive.jobfolder.values():
      good = not value.is_tagged
      if good: break
  if not good:
    print '**** Current job-folders and sub-folders are all off; '             \
          'jobparams (except onoff) and collect will not work.'
    return
  if interactive.jobfolder_path is None: return
  dir = join( splitpath(interactive.jobfolder_path)[0], 
              interactive.jobfolder.name[1:] ) 
  if exists(dir): chdir(dir)
  else: 
    print "In {0}, but no corresponding directory on disk."                    \
          .format(interactive.jobfolder.name.split('/')[-2])
  return


def iterate(self, event):
  """ Goes to next (untagged) job. """
  from pylada import interactive
  if interactive.jobfolder is None: return

  args = event.split()
  if len(args) > 1: 
    print "Invalid argument {0}.".format(event)
    return
  elif len(args) == 0:
    if "_pylada_subjob_iterator" in interactive.__dict__:
      iterator = interactive._pylada_subjob_iterator
    else:
      iterator = interactive.jobfolder.root.itervalues()
    while True:
      try: job = iterator.next()
      except StopIteration: 
        print "Reached end of job list."
        return 
      if job.is_tagged: continue
      break
    interactive._pylada_subjob_iterator = iterator
    if "_pylada_subjob_iterated" not in interactive.__dict__:
      interactive._pylada_subjob_iterated = []
    interactive._pylada_subjob_iterated.append(interactive.jobfolder.name)
    goto(self, job.name)
    print "In job ", interactive.jobfolder.name
  elif args[0] == "reset" or args[0] == "restart":
    # remove any prior iterator stuff.
    interactive.__dict__.pop("_pylada_subjob_iterator", None)
    interactive.__dict__.pop("_pylada_subjob_iterated", None)
    print "In job ", interactive.jobfolder.name
  elif args[0] == "back" or args[0] == "previous":
    if "_pylada_subjob_iterated" not in interactive.__dict__:
      print "No previous job to go to. "
    else:
      goto(self, interactive._pylada_subjob_iterated.pop(-1))
      if len(interactive._pylada_subjob_iterated) == 0:
        del interactive._pylada_subjob_iterated
      print "In job ", interactive.jobfolder.name


def completer(self, event):
  from IPython import TryNext
  from pylada import interactive
  if len(event.line.split()) > 2: raise TryNext()

  if '/' in event.symbol:
    subkey = ""
    for key in event.symbol.split('/')[:-1]: subkey += key + "/"
    try: subdict = interactive.jobfolder[subkey]
    except KeyError: raise TryNext()
    if hasattr(subdict, "children"): 
      if hasattr(subdict.children, "keys"):
        return [subkey + a + "/" for a in subdict.children.keys()]
    raise TryNext()
  else:
    result = [a + "/" for a in interactive.jobfolder.children.keys()]
    result.extend(["/", "next", "reset"])
    if interactive.jobfolder.parent is not None: result.append("../")
    if len(getattr(interactive, "_pylada_subjob_iterated", [])) != 0:
      result.append("previous")
    return result

