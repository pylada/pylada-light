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

def test():
  from random import randint
  from tempfile import mkdtemp
  from shutil import rmtree
  from os import makedirs
  from os.path import exists, join
  from copy import deepcopy
  from pylada.jobfolder import JobFolder, JobParams
  from dummy import functional

  # create jodictionary to explore
  root = JobFolder()
  for type, trial, size in [('this', 0, 10), ('this', 1, 15), ('that', 2, 20), ('that', 1, 20)]:
    jobfolder = root / type / str(trial)
    jobfolder.functional = functional
    jobfolder.params['indiv'] = size
    if type == 'that': jobfolder.params['value'] = True

  # checks that attributes are forwarded correctly
  jobparams = JobParams(root)
  assert len(jobparams.functional.values()) == 4
  for i, (name, value) in enumerate(jobparams.functional.iteritems()):
    assert repr(value) == repr(functional)
  assert i == 3
  for i, (name, value) in enumerate(jobparams.indiv.iteritems()):
    if   name == '/this/0/': assert value == 10
    elif name == '/this/1/': assert value == 15
    elif name == '/that/1/': assert value == 20
    elif name == '/that/2/': assert value == 20
    else: raise RuntimeError()
  assert i == 3

  # checks that attributes only a few posses are forwarded
  for i, (name, value) in enumerate(jobparams.value.iteritems()):
    if   name == '/that/1/': assert value == True
    elif name == '/that/2/': assert value == True
    else: raise RuntimeError()
  assert i == 1
  
  # check that an AttributeError is raised on an unknown attribute
  try: jobparams.that
  except AttributeError: pass
  else: raise RuntimeError()

  # check wildcard indexing.
  jobparams.unix_re = True
  def check_items(regex, keys, d):
    i = 0
    for name in d[regex].iterkeys():
      i += 1
      assert name in keys, KeyError((regex, name))
    assert i == len(keys), RuntimeError(regex)
  check_items('*/1', set(['/this/1/', '/that/1/']), jobparams)
  check_items('this', set(['/this/1/', '/this/0/']), jobparams)
  check_items('that/2/', set(['/that/2/']),jobparams)
  
  # check adding an item
  jobfolder = JobFolder()
  jobfolder.functional = functional
  jobfolder.params['indiv'] = 25
  jobfolder.params['value'] = 5
  jobfolder.params['another'] = 6
  jobparams['/this/0/another'] = jobfolder

  # continue wildcard indexing.
  check_items('*/*/another', ['/this/0/another/'], jobparams)
  check_items('*/*/another/', ['/this/0/another/'], jobparams)
  check_items('*/another', [], jobparams)
  check_items('../that', ['/that/2/', '/that/1/'], jobparams['this'])
  check_items('../that', [], jobparams['this/0'])
  check_items('../*', ['/this/0/', '/this/1/', '/this/0/another/'], jobparams['this/0'])
  check_items('../*', ['/this/0/', '/this/1/', '/this/0/another/'], jobparams['this/*'])

  # check regex indexing.
  jobparams.unix_re = False
  check_items('.*/1', set(['/this/1/', '/that/1/']), jobparams)
  check_items('this', set(['/this/1/', '/this/0/', '/this/0/another/']), jobparams)
  check_items('that/2/', set(['/that/2/']), jobparams)
  check_items('.*/.*/another', ['/this/0/another/'], jobparams)
  check_items('.*/another', ['/this/0/another/'], jobparams)

  jobparams.unix_re = True
  jobparams.naked_end = True
  assert isinstance(jobparams.another, int)
  for i, (key, value) in enumerate(jobparams['*/1'].indiv.iteritems()):
    assert {'/this/0/': 10, '/this/1/': 15, '/that/1/': 20,
            '/that/2/': 20, '/this/0/another/': 25}[key] == value
  assert i == 1

  # check setting attributes.
  jobparams.indiv = 5
  for i, (key, value) in enumerate(jobparams.indiv.iteritems()): assert value == 5
  assert i == 4
  jobparams['*/0'].indiv = 6
  assert len(jobparams.indiv.values()) == 5
  for i, (key, value) in enumerate(jobparams.indiv.iteritems()): 
    assert value == (6 if '0' in key else 5)
  assert i == 4

  # resets attributes for later tests. 
  for key in jobparams.keys():
    jobparams[key].indiv = {'/this/0/': 10, '/this/1/': 15, '/that/1/': 20,\
                            '/that/2/': 20, '/this/0/another/': 25}[key]
  jobfolder = deepcopy(jobparams.jobfolder)

  # check deleting.
  del jobparams['/*/1']
  assert '/this/1/' not in jobparams and '/that/1/' not in jobparams\
         and '/that/2/' in jobparams and '/this/0/' in jobparams\
         and '/this/0/another/' in jobparams

  # check concatenate with job-folder.
  jobparams.indiv = 2
  jobparams.concatenate(jobfolder)
  for i, (name, value) in enumerate(jobparams.functional.iteritems()):
    assert repr(value) == repr(functional)
  assert i == 4
  i = 0
  for i, (key, value) in enumerate(jobparams.indiv.iteritems()):
    assert {'/this/0/': 10, '/this/1/': 15, '/that/1/': 20,
            '/that/2/': 20, '/this/0/another/': 25}[key] == value,\
           Exception((key, value))
  assert i == 4

  # check concatenate with Jobparams and indexing.
  del jobparams['/*/1']
  jobparams.indiv = 2
  jobparams.concatenate(JobParams(jobfolder)['/*/1'])
  for i, (name, value) in enumerate(jobparams.functional.iteritems()):
    assert repr(value) == repr(functional)
  assert i == 4
  i = 0
  for i, (key, value) in enumerate(jobparams.indiv.iteritems()):
    assert {'/this/0/': 2, '/this/1/': 15, '/that/1/': 20,
            '/that/2/': 2, '/this/0/another/': 2}[key] == value,\
           Exception((key, value))
  assert i == 4

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 1: path.extend(argv[1:])
  test()
