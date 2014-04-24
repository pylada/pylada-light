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

# to be run from a compound directory, (so ../input_generic works, etc).
import os, os.path
from pylada.vasp.nlep.mpopt import *

#from pylada.opt import read_input
from pylada.vasp import read_input
from pylada.vasp.extract import Extract, ExtractGW
from pylada.vasp.specie import U, nlep
from pylada.vasp import Vasp
# Things which need to be put into dictionary of input script.
#global_dict = { "Extract": Extract, "ExtractGW": ExtractGW, "Vasp": Vasp, "nlep": nlep, "U": U }


def monitor3(fname, fout):
    import os,sys,fileinput
    from math import sqrt

    # there are how many words of respective lines constitute label, not data
    calc_hdr = 8
    ret_hdr = 5
    # this accounts for fact that output now inclues RMS^2 and RMS as last to items on "returning..."
    # for now, recalc them anyway (check same).
    ret_ftr = 2

    iter = 0
    for ln in file(fname):
        ln = ln.split()
        if (len(ln) > 2 and ln[2] == "calculating"):
            savex = [float(ln[j]) for j in range(calc_hdr,len(ln))]
            str = "%d    " % iter
            for xi in savex:
                str += "%.12e " % xi
        if (len(ln) > 2 and ln[2] == "returning"):
    #        print "line: ", ln
            sum = 0
    #        s = "%d   %.12e %.12e %.12e  "  % (iter, savex, savey, savez)
            str += "    "
            for i in range(ret_hdr, len(ln)-ret_ftr):
                term = ln[i]
    #            print "term: ", term
                term = term.strip('][')
                if (term != ""):
    #                print "term2: ", term
                    val = float(term)
                    sum += val*val
    #                str += " %.8e " % val
            str += "    %.12e"  % sqrt(sum)
            fout.write("%s\n" % str)
            iter += 1
    


def get_nlep(vasp):
  """ Returns vector of parameters from L{vasp} attribute. """
  from numpy import array
  result = []
  for symbol, specie in vasp.species.items():
    for nlep_params in specie.U:
      #        print "nlep_param = ", nlep_params
      if nlep_params["func"] == "nlep": 
        result.append(nlep_params["U"]) 
      elif nlep_params["func"] == "enlep": 
        result.append(nlep_params["U0"]) # first energy
        result.append(nlep_params["U1"]) # second energy
  return array(result, dtype="float64")


def load_test(run, job=None, rank=None, system=None, files_dest_dir=None, getdir=None):
  import shutil
  import os
  import os.path
  from pylada.vasp.extract import Extract

  generic_input = load_run_input()
  if (getdir == None):
      getdir = generic_input.indir
  dir = os.getcwd()
  dir += "/%s/" % (getdir)
  if (job != None):
    dir = dir + "job%d/" % job
  if (rank != None):
    dir = dir + "rank%d/" % rank
  if (system != None):
    dir = dir + "%s/" % system
  dir = dir + "/%d" % run

  print "extracting vasp object from: ", dir
  test = Extract(outcar=dir)
#  nlep = get_nlep(test.vasp)
  nlep = get_nlep(test.functional)

  if (files_dest_dir != None):
    if (not os.path.isdir(files_dest_dir)):
      os.mkdir(files_dest_dir)
    prefix = ""
    if (system != None):
      prefix = "%s_" % system
    shutil.copy("%s/OUTCAR" % dir, "%s/%sOUTCAR" % (files_dest_dir, prefix))
    shutil.copy("%s/INCAR" % dir, "%s/%sINCAR" % (files_dest_dir, prefix))
    shutil.copy("%s/POSCAR" % dir, "%s/%sPOSCAR" % (files_dest_dir, prefix))

  return test, nlep


def pretty_print_one(res, tag, input, test):
  import numpy as np
  import math
  print "%s ERRORS::" % (tag)
  nbands = len(input.nlep_band_idx)
  if (nbands == 6):
    print "evals:  deep-d     VBM3     VBM2     VBM1    CBM1     CBM2"
  else:
    print "evals:   VBM3     VBM2     VBM1    CBM1     CBM2"
  specialk = ["G", "X", "L"]
  for kidx in range(0,3):
    s = "%s   " % specialk[kidx]
    for bidx in range(0,nbands):
      idx = nbands*kidx + bidx
      s += "%f  " % (res[idx])
    print s
  #     deep-d   VBM3  VBM2  VBM1  CBM1 CBM2   
  # G:
  # X:
  # L:

  baseidx = 3 * nbands
  print "charges:   s      p      d"

  for aidx in range(0,2):
    sp = input.species_names[input.atom_idx[aidx]]
    s = "%s   " % sp
    for cidx in range(0,3):
      idx = baseidx + 3*input.atom_idx[aidx] + cidx
      s += "%f  " % (res[idx])
    print s

  baseidx += 2 * 3
  print "pressure: %f   " % res[baseidx]

  res = np.array(res)
  rms = math.sqrt(sum(res*res))
  print "%s RMS = " % (tag), rms
  return rms
  
def pretty_print(res, res_unweighted, input, test, nlep):
  rms = pretty_print_one(res, "WEIGHTED ", input, test)
  raw_rms = pretty_print_one(res_unweighted, "UNWEIGHTED ", input, test)

  print "NLEP parameters"
#  for symbol, specie in test.vasp.species.items():
  for symbol, specie in test.functional.species.items():
    for nlep_params in specie.U:
      print symbol, nlep_params

  return rms, raw_rms

def find_last(sysname, job, withranks=False):
    import fileinput
    generic_input = load_run_input()
    if (withranks):
        rankstr = "rank0/"
    else:
        rankstr = ""
    lastln = ""

    evals_file = "%s/job%d/%s/%s/evals" % (generic_input.indir, job, rankstr, sysname)
    run = 0
    for ln in fileinput.input(evals_file):
        # each line containing "returning from array_function" indicates a successful vasp run. we count these
        if ln.find("returning from array_function") >= 0:
            run += 1
    if (run > 0):
        run -= 1  # seems to be needed ?
    return run


def find_best(system_names, njobs=16, withranks=False, getdir=None):
  import fileinput
  from operator import itemgetter
  from boost.mpi import world

  print "find best, jobs, ranks = ", njobs, withranks
  generic_input = load_run_input()
  if getdir == None:
      getdir = generic_input.indir
  all_evals = file("all_evals", "w")
  if (withranks):
    rankstr = "rank0/"
  else:
    rankstr = ""
  if (njobs == -1):
      njobs = world.size
  for p in range (0,njobs):
    evals_file = "%s/job%d/%s/evals" % (getdir, p, rankstr)
    evals = monitor3(evals_file, all_evals)
    all_evals.write("\n")
  all_evals.close()

  lastln = ""
  all_solns = []
  jobidx = 0
  for ln in fileinput.input("all_evals"):
    ln = ln.strip()
    if (ln == ""):
      # last line is last iter of a job
#      print lastln
      dat = [jobidx] + [float(x) for x in lastln.split()]
      all_solns.append(dat)
      jobidx += 1
    lastln = ln
    
  rmsidx = len(all_solns[0])-1
  sorted_solns = sorted(all_solns, key=itemgetter(rmsidx))
  for s in  sorted_solns:
    print s

  job = int(sorted_solns[0][0])
  run = int(sorted_solns[0][1])
  runs = {}
  for s in system_names:
      runs[s] = find_last(s, job, withranks)
  return job, runs
  
def get_analog_name(cat,an):
  analogs = {'Al':'Mg', 'Mg':'Al',  'Ga':'Zn', 'Zn':'Ga',  'In':'Cd', 'Cd':'In',  
             'N':'O', 'O':'N',  'P':'S', 'S':'P',   'As':'Se', 'Se':'As'}
  sys = "%s%s" % (analogs[cat], analogs[an])
  return sys

def prepare_analog_fit(test, nlep_params):
  print "preparing initial conditions for analogous elements"
  analogs = {'Al':'Mg', 'Mg':'Al',  'Ga':'Zn', 'Zn':'Ga',  'In':'Cd', 'Cd':'In',  
             'N':'O', 'O':'N',  'P':'S', 'S':'P',   'As':'Se', 'Se':'As'}
  adict = {}
  for symbol, specie in test.functional.species.items():
      row = []
      for nlep_params in specie.U:
          print symbol, analogs[symbol], nlep_params
          row.append(nlep_params)
      adict[analogs[symbol]] = row
  return adict
