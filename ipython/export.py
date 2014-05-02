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

""" IPython export magic function. """
__docformat__ = "restructuredtext en"


def export(self, event):
  """ Tars files from a calculation.  """
  import argparse
  import tarfile
  from os import getcwd
  from os.path import exists, isfile, extsep, relpath, dirname, join
  from glob import iglob
  from ..misc import RelativePath
  from .. import interactive
  from . import get_shell
  shell = get_shell(self)

  parser = argparse.ArgumentParser(prog='%export',
      description='Exports input/output files from current job-folder. '       \
                  'Depending on the extension of FILE, this will create '      \
                  'a simple tar file, or a compressed tar file. Using the '    \
                  'option --list, one can also obtain a list of all files '    \
                  'which would go in the tar archive. '                        \
                  'Finally, this function only requires the \"collect\" '      \
                  'exists in the usernamespace. It may have been declared '    \
                  'from loading a job-folder using \"explore\", or directly '  \
                  'with \"collect = vasp.MassExtract()\".' )
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument( '--list', action="store_true", dest="aslist",
                       help='Do not tar, return a list of all the files.' )
  group.add_argument( 'filename', metavar='FILE', type=str,
      default='export.tar.gz', nargs='?',
      help='Path to the tarfile. Suffixes ".gz" and ".tgz" indicate '          \
           'gzip compression, whereas ".bz" and ".bz2" indicate bzip '         \
           'compression. Otherwise, no compression is used.')
  parser.add_argument( '--input', action="store_true", dest="input",
                       help='Include input (INCAR/crystal.d12) files.' )
  parser.add_argument( '--dos', action="store_true", dest="dos",
                       help='Include Density of States (DOSCAR) files.' )
  parser.add_argument( '--structure', action="store_true", dest="structure",
                       help='Include structure input (POSCAR) files.' )
  parser.add_argument( '--charge', action="store_true", dest="charge",
                       help='Include charge (CHGCAR) files.' )
  parser.add_argument( '--contcar', action="store_true", dest="contcar",
                       help='Include CONTCAR files.' )
  parser.add_argument( '--wavefunctions', action="store_true", dest="wavefunctions",
                       help='Include wavefunctions (WAVECAR/crystal.98) files.' )
  parser.add_argument( '--procar', action="store_true", dest="procar",
                       help='Include PROCAR files.' )
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument( '--down', action="store_true", dest="down",
                       help='Tar from one directory down.' )
  group.add_argument( '--from', type=str, dest="dir", default=None,
                       help='Root directory from which to give filenames. '    \
                            'Defaults to current working directory.' )
  group.add_argument( '--with', type=str, dest="others", nargs='*',
      help='Adds pattern or filename to files to export.'                      \
           'Any file in any visited directory matching the given pattern '     \
           'will be added to the archive. This options can be given more than '\
           'once if different file patterns are required.' )

  try: args = parser.parse_args(event.split())
  except SystemExit as e: return None

  collect = shell.user_ns.get('collect', None)
  rootpath = getattr(collect, 'rootpath', None)
  if collect is None:
    print "Could not find 'collect' object in user namespace."
    print "Please load a job-dictionary."
    return

  kwargs = args.__dict__.copy()
  kwargs.pop('filename', None)

  if rootpath is None: 
    if hasattr(shell.user_ns.get('collect', None), 'rootpath'): 
      rootpath = shell.user_ns.get('collect').rootpath
  directory = getcwd() if rootpath is None else dirname(rootpath)

  if args.down: directory = join(directory, '..')
  elif args.dir is not None: directory = RelativePath(args.dir).path

  # set of directories visited.
  directories = set()
  # set of files to tar
  allfiles = set()
  for file in collect.iterfiles(**kwargs):
    allfiles.add(file)
    directories.add(dirname(file))
  # adds files from "with" argument.
  if hasattr(args.others, "__iter__"):
    for pattern in args.others:
      for dir in directories:
        for sfile in iglob(join(dir, pattern)):
          if exists(sfile) and isfile(sfile): allfiles.add(sfile)
  # adds current job folder.
  if interactive.jobfolder_path is not None:
    if isfile(interactive.jobfolder_path):
      allfiles.add(interactive.jobfolder_path)

  # now tar or list files.
  if args.aslist:
    from IPython.utils.text import SList
    directory = getcwd()
    return SList([relpath(file, directory) for file in allfiles])
  else:
    # get filename of tarfile.
    args.filename = relpath(RelativePath(args.filename).path, getcwd())
    if exists(args.filename): 
      if not isfile(args.filename):
        print "{0} exists but is not a file. Aborting.".format(args.filename)
        return 
      a = ''
      while a not in ['n', 'y']:
        a = raw_input( "File {0} already exists.\nOverwrite? [y/n] "           \
                       .format(args.filename) )
      if a == 'n': print "Aborted."; return

    # figure out the type of the tarfile.
    if args.filename.find(extsep) == -1: endname = ''
    else: endname = args.filename[-args.filename[::-1].find(extsep)-1:][1:]
    if endname in ['gz', 'tgz']:   tarme = tarfile.open(args.filename, 'w:gz')
    elif endname in ['bz', 'bz2']: tarme = tarfile.open(args.filename, 'w:bz2')
    else:                          tarme = tarfile.open(args.filename, 'w')
    for file in allfiles:
      tarme.add(file, arcname=relpath(file, directory))
    tarme.close()
    print "Saved archive to {0}.".format(args.filename)

def completer(self, event):
  """ Completer for export. """
  from glob import iglob
  from itertools import chain
  from os.path import isdir
  from IPython.core.completer import expand_user, compress_user
  data = event.line.split()
  if    (len(event.symbol) == 0 and data[-1] == "--from") \
     or (len(event.symbol) > 0  and data[-2] == "--from"):
    relpath, tilde_expand, tilde_val = expand_user(data[-1])
    dirs = [f.replace('\\','/') + "/" for f in iglob(relpath+'*') if isdir(f)]
    return [compress_user(p, tilde_expand, tilde_val) for p in dirs]

  if    (len(event.symbol) == 0 and len(data) > 0 and data[-1] == "--with")    \
     or (len(event.symbol) > 0  and len(data) > 1 and data[-2] == "--with"):
     return []

  data = set(data) - set(["export", "%export"])
  result = set( [ '--incar', '--doscar', '--poscar', '--chgcar', '--contcar', 
                  '--potcar', '--wavecar', '--procar', '--list', '--down',
                  '--from', '--with' ] )


  if '--list' not in data:
    other = event.line.split()
    if '--from' in other:
      i = other.index('--from')
      if i + 1 < len(other): other.pop(i+1)
    other = [ u for u in (set(other)-result-set(['export', '%export']))        \
              if u[0] != '-' ]
    if len(other) == 0:
      for file in chain( iglob('*.tar'), iglob('*.tar.gz'), 
                         iglob('*.tgz'), iglob('*.bz'), iglob('*.bz2') ):
        result.add(file)
      relpath, tilde_expand, tilde_val = expand_user(data[-1])
      result |= set( [ f.replace('\\','/') + "/" for f in iglob(relpath+'*')   \
                       if isdir(f) ] )
    elif len(other) == 1 and len(event.symbol) != 0: 
      result.discard('--list')
      other = event.symbol
      if '.' in other: other = other[:other.find('.')]
      string = "{0}*.tar {0}*.tar.gz {0}*.tgz {0}*.tar.bz"                     \
               "{0}*.tar.bz2 {0}*/".format(other)
      result |= set([u for u in iglob(string)])
      if isdir(other) and other[-1] != '/':
        string = "{0}/*.tar {0}/*.tar.gz {0}/*.tgz {0}/*.tar.bz "              \
                 "{0}/*.tar.bz2 {0}*/".format(other)
        result |= set([u for u in iglob(string)])

  result = result - data 
  if '--down' in data: result.discard('--from')
  if '--from' in data: result.discard('--down')
  return list(result) + ['--with']
