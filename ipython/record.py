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

""" Allows to rapidly store and reload information to disk. 

    After a lengthy calculation it may be practical to save the results to file. 

    >>> result = 5 
    >>> record result

    The above save the variable ``result`` to disk.
    It can then be retrieved, at the expense of overwriting the value in the
    user namespace.

    >>> result
    5
    >>> result = 8
    >>> record --load result
    Reloaded result(=5) from record .pylada_record.
    >>> result
    5

    All records can be viewed and listed using:

    >>> record --view
    result = 5
    >>> record --list
    ['result']
"""
def record(self, cmdl):
  """ Records variables to file. """
  from argparse import ArgumentParser
  from types import ModuleType
  from pylada.record import Record
  
  parser = ArgumentParser( prog="%record",
                           description="Allows to rapidly store and reload information to disk.")
  parser.add_argument( '--file', dest="filename", default=".pylada_record", type=str, 
                       help="Name of file where variables are recorded." )
  parser.add_argument( 'vars', metavar='VAR', type=str, nargs='*',
                       help='Name of the variable(s) to record or remove.' )
  parser.add_argument( '--namespace', '-n', type=str, 
                       help='Reloads records into a specific namespace'
                            'rather than the global one.' )
  group = parser.add_mutually_exclusive_group()
  group.add_argument( '--remove', action="store_true", dest='remove',
                       help="Removes variable from record if present." )
  group.add_argument( '--update', action="store_true", dest='update',
                       help="Adds/updates variables to record. Default." )
  group.add_argument( '--list', action="store_true", dest='list',
                       help="Lists objects in record." )
  group.add_argument( '--view', action="store_true", dest='view',
                       help="View a given record, if possible printable."
                            "If not VAR argument is given, all records are listed." )
  group.add_argument( '--load', action="store_true", dest='load',
                      help="Reloads variables from record into user namespace."\
                           "If no VAR are given, reload all records." )


  try: args = parser.parse_args(cmdl.split())
  except SystemExit:
    if '-h' in cmdl: print __doc__[__doc__.find('\n'):].replace('\n    ', '\n')
    return

  if not (args.remove or args.list or args.load or args.view): args.update = True
  if args.namespace is not None and not args.load: 
    parser.print_usage()
    print "\n--namespace argument has no effect except when reloading record."
    return
  if len(args.vars) == 0 and not (args.list or args.load or args.view):
    parser.print_usage()
    print "\n*****************\n"\
          "At least on VAR argument required.\n"\
          "*****************"
    return 

  # open record file.
  record = Record(args.filename)

  attributes = dir(record)
  if args.list: return attributes
  if args.view: 
    if len(args.vars) == 0: args.vars = attributes
    for var in args.vars:
      if var not in attributes:
        print "Could not find {0} in {1}.".format(var, args.filename)
        continue
      try: string = str(getattr(record, var))
      except: 
        print "{0} in record {1} is not printable.".format(var, args.filename)
        continue
      else:
        if len(string) > 30: string = string[:25] + "..."
        if "\n" in string: string = string[:string.find('\n')] + "..."
        print "{0} = {1}".format(var, string)
  elif args.remove: 
    # Remove argumenst from record.
    for var in set(args.vars):
      if var not in attributes: 
        print "{0} could not be found in record file {1}.".format(var, args.filename)
        continue
      value = getattr(record, var)
      delattr(record, var)
      try: string = str(value)
      except: print "Removing {0} from record {1}.".format(var, args.filename)
      else:
        if len(string) > 30: string = string[:25] + "..."
        if "\n" in string: string = string[:string.find('\n')] + "..."
        print "Removing {0}(={1}) from record {2}".format(var, string, args.filename)
  elif args.load: 
    if len(args.vars) == 0: args.vars = attributes
    if args.namespace is None: namespace = self.api.user_ns
    elif args.namespace in self.api.user_ns:
      namespace = self.api.user_ns[args.namespace]
      if not isinstance(namespace, dict):
        if not hasattr(namespace, '__dict__'):
          print "{0} is not a namespace or a dictionary.".format(args.namespace)
        namespace = namespace.__dict__
    else:
      self.api.user_ns[args.namespace] = ModuleType(name=args.namespace)
      namespace = self.api.user_ns[args.namespace].__dict__
    allattr = record._allattr()
    for key in args.vars:
      if key not in allattr:
        print "Could not find {0} in {1}.".format(key, args.filename)
        continue
      namespace[key] = getattr(record, key)
      try:string = str(namespace[key])
      except: print "Reloaded {0} from record {1}.".format(key, args.filename)
      else:
        if len(string) > 30: string = string[:25] + "..."
        if "\n" in string: string = string[:string.find('\n')] + "..."
        print "Reloaded {0}(={2}) from record {1}.".format(key, args.filename, string)
    return
  else:
    # Add arguments to record.
    for key in set(args.vars): 
      # checks value exists in user namespace.
      if key not in self.api.user_ns: 
        print "Could not find {0} in user namespace.".format(key)
        continue
      # checks the argument can be pickled.
      setattr(record, key, self.api.user_ns[key])
         
def completer(self, event): 
  """ Completer for %record magic function. """ 
  from pylada.opt import RelativeDirectory
  from pickle import load
  from os.path import isdir, exists

  result = []
  data = event.line.split()[1:]
  if    (len(event.symbol) == 0 and len(data) > 0 and data[-1] == "--n") \
     or (len(event.symbol) > 0  and len(data) > 1 and data[-2] == "--n") \
     or (len(event.symbol) == 0 and len(data) > 0 and data[-1] == "--namespace") \
     or (len(event.symbol) > 0  and len(data) > 1 and data[-2] == "--namespace"):
   return [ key for key, value in self.api.user_ns.iteritems() \
            if key[0] != '_' and hasattr(value, '__dict__') ]

  if    (len(event.symbol) == 0 and len(data) > 0 and data[-1] == "--file") \
     or (len(event.symbol) > 0  and len(data) > 1 and data[-2] == "--file"):
    other = event.symbol
    string = '%mglob "cont:This is a record." {0}*'.format(other)
    result = [u for u in self.api.magic(string)]
    string = '%mglob dir:{0}*'.format(other)
    result.extend([u for u in self.api.magic(string)])
    if isdir(other) and other[-1] != '/':
      string = '%mglob "cont:This is a record." {0}/*'.format(other)
      result.extend([u for u in self.api.magic(string)])
    return result

  if '--file' not in data: result.append('--file') 
  options = set(['--list', '--view', '--load', '--remove', '--update'])
  if '-n' in data or '--namespace' in data and '--load' not in data:  result.append('--load') 
  elif '--load' in data and '-n' not in data and '--namespace' not in data: 
    result.append('--namespace')
  elif len(options.intersection(set(data))) == 0:
    result.extend(['--list', '--view', '--load', '--remove', '--namespace', '--update'])
  if len(set(['--load', '--remove', '--view']).intersection(set(data))) != 0:
    path = '.pylada_record'
    known = [u for u in data]
    if '--file' in data: 
      index = data.index('--file')
      assert len(data) > index + 1
      path = RelativeDirectory(data[index+1]).path
      known.pop(index + 1) 
    if exists(path): 
      with open(path) as file: keys = load(file)
      if '--namespace' in known:
        index = known.index('--namespace')
        if len(known) > index + 1: known.pop(index+1)
      if '-n' in known:
        index = known.index('-n')
        if len(known) > index + 1: known.pop(index+1)
      result.extend(keys - set(known))
  else:
    result.extend([u for u in self.api.user_ns.iterkeys() if u[0] != '_'])
  return result

