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


def get_result_size(sys_input, run_input):
  f0 = run_input.scale_eigenvalues 
  f1 = run_input.scale_partial_charges
  f2 = run_input.scale_occupations
  f3 = run_input.scale_pressure
  size = 0
  if f0 > 1e-12: size += len(run_input.k_idx) * len(sys_input.nlep_band_idx)
  if f1 > 1e-12: size += len(run_input.pc_orbital_idx) * len(sys_input.vasp.species)   
  if f2 > 1e-12: size += len(run_input.k_idx) * len(sys_input.nlep_band_idx)
  if f3 > 1e-12: size += 1
  return size


def test_objective(args, sys_input):
#  eigs, pc, pressure, occs = test_objective(args, sys_input)
  print "test_objective called: ", args
#  eigs = (1.0 + args[0]) * sys_input.gw_in.qp_eigenvalues
  eigs = sys_input.gw_in.qp_eigenvalues
  pc = sys_input.dft_in.partial_charges
  pressure = 0
  occs = sys_input.gw_in.occupations
  return eigs,pc,pressure,occs

def wprint(*args):
  from boost.mpi import world
  s = "rank " + str(world.rank)
  for a in args:
    s = s + " " + str(a)
  print s

def array_function_calcs(x, raweigs, rawpc, pressure, rawoccs, sys_input, run_input, special_pc=None, ignore_weights=False):    
  from boost.mpi import world
  import numpy as np
  import math
  from quantities import eV

  if special_pc != None:
    wprint ("using pc overrides")
    sys_input_pc = special_pc
  else:
    sys_input_pc = sys_input.dft_in.partial_charges
    
  k_idx = run_input.k_idx
  nlep_band_idx = sys_input.nlep_band_idx
  gw_band_idx = sys_input.gw_band_idx

  target_eigs = [sys_input.gw_in.qp_eigenvalues[k_idx[j]][gw_band_idx[i]] for j in range (0,len(k_idx)) for i in range(0,len(gw_band_idx))]
  target_eigs = np.array([v.rescale(eV) for v in target_eigs])
  eigs = np.array([raweigs[k_idx[j]][nlep_band_idx[i]] for j in range (0,len(k_idx)) for i in range(0,len(nlep_band_idx))])
  pc = np.array([rawpc[sys_input.atom_idx[i]][run_input.pc_orbital_idx[k]] for i in range(0,len(sys_input.vasp.species)) for k in range(0,len(run_input.pc_orbital_idx))])
  target_pc = np.array([sys_input_pc[i][run_input.pc_orbital_idx[k]] for i in range(0,len(sys_input.vasp.species)) for k in range(0,len(run_input.pc_orbital_idx))])

  if (not run_input.floating_vbm):
    band_shift = raweigs[run_input.gamma_point_idx][sys_input.nlep_valence_band_idx] - sys_input.gw_in.qp_eigenvalues[run_input.gamma_point_idx][sys_input.gw_valence_band_idx]
    band_shift = float(band_shift.rescale(eV))
  else:
    eigdelta = (eigs - target_eigs)
    if (ignore_weights):
      band_shift = sum(eigdelta)/float(len(eigdelta))
    else:
      eigdelta = eigdelta * sys_input.eigenvalue_weights.flat
      normalizer = float(sum(sys_input.eigenvalue_weights.flat))
      band_shift = sum(eigdelta)/normalizer
#  if (sys_input.cation == "Mg"):
#    print "total Mg hack!"
#    target_pc[1] -= 6.

  target_occs = [sys_input.gw_in.occupations[k_idx[j]][gw_band_idx[i]] for j in range (0,len(k_idx)) for i in range(0,len(gw_band_idx))]
  target_occs = np.array([v.rescale(eV) for v in target_occs])
  occs = np.array([rawoccs[k_idx[j]][nlep_band_idx[i]] for j in range (0,len(k_idx)) for i in range(0,len(nlep_band_idx))])

  wprint ("eigs = ", eigs)
  wprint ("target_eigs = ", target_eigs)
  wprint ("pc = ", pc)
  wprint ("target_pc = ", target_pc)
  wprint ("pressure ", pressure)
  wprint ("occs = ", occs)
  wprint ("target_occs", target_occs)
  wprint ("band_shift = ", band_shift, run_input.floating_vbm)
  wprint ("eigenvalue weights = ", sys_input.eigenvalue_weights)

  eigs = (eigs - target_eigs) - band_shift
  if (not ignore_weights):
    eigs = eigs * sys_input.eigenvalue_weights.flat
  pc = (pc - target_pc)
  if (not ignore_weights):
    pc = pc * sys_input.pc_orbital_weights.flat
  occs = (occs - target_occs)
  
  # note, no bounds checking; we already checked
  if (ignore_weights):
    f0 = 1.0
    f1 = 1.0
    f2 = 0.0
    f3 = 1.0
    wprint ("WARNING: array function calcs ignoring weights")
  else:
    f0 = run_input.scale_eigenvalues 
    f1 = run_input.scale_partial_charges
    f2 = run_input.scale_occupations
    if (sys_input.scale_pressure == None):
      f3 = run_input.scale_pressure
    else:
      f3 = sys_input.scale_pressure
  # Note, we need to have at least nparam criteria to use MINPACK,
  # which is what leastsq is using]:
  
  wprint ("pressure scaled by ", f3)

  size = get_result_size(sys_input, run_input)
  result = np.zeros( (size,), dtype = "float64" )
  index = 0
  if f0 > 1e-12:
    result[index:len(eigs)] = (eigs * f0).flat
    index += len(eigs)
  if f1 > 1e-12:
#    result[index:pc.size+index] = (pc / math.sqrt(float(pc.size)) * f1).flat    # why sqrt??
    result[index:pc.size+index] = (pc * f1).flat
    index += pc.size
  if f2 > 1e-12:
    result[index:occs.size+index] = (occs * f2).flat
    index += occs.size  
  if f3 > 1e-12:
    result[index] = pressure * f3

### NOTE: normalization by, e.g. how many eigenvalues are of interest not longer in code.  just relying on hand-specified weights from Stephan L.

  return result

def check_bounds(args, sys_input, run_input, x0=None, ranges=None):
  # keep parameters within bounds through a quadratic penalty term 
  from boost.mpi import world
  import numpy as np

  if (ranges == None):
    ranges = [sys_input.bounds[1] for i in range(0,len(args))]
  if (x0 == None):
    x0 = args
  bounds = [x0-ranges, x0+ranges]
  wprint ("in check_bounds: ranges ", ranges, " bounds ", bounds, " sys_input.bounds ", sys_input.bounds)

  if (run_input.bounds_penalty > 0):
    penalty = 0
    for i in range(0,len(args)):
      a = args[i]
      if (a < bounds[0][i]):
        pterm = run_input.bounds_penalty * ((a - bounds[0][i])**2)
        wprint ("rank %d, arg %d, penalizing %e by %e" % (world.rank, i, a, pterm))
        penalty += pterm
      if (a > bounds[1][i]):
        pterm = run_input.bounds_penalty * ((a - bounds[1][i])**2)
        wprint ("rank %d, arg %d, penalizing %e by %e" % (world.rank, i, a, pterm))
        penalty += pterm
    # MINPACK wants the actual array of errors.  How to distribute the penalty?
    # just add to all errors:
    if (penalty > 1):  # note this allows small violations, but this way these get evaluated and smoothly pushed back
      # Candidates that fall into penalty <= 1 will be evaluated by VASP
      # if we use "penalty < 0" system finds (false) optima just over boundary
      size = get_result_size(sys_input, run_input)
      result = np.zeros( (size,), dtype = "float64" )
      for i in range(0,len(result)):
        result[i] = penalty
      return result
    else:
      return None  # bounds ok
  

def report_result(args, result, msg, evals_file):
  from boost.mpi import world
  import math
  s = "rank %d calculating fit to data at x= " % (world.rank)
  for a in args:
    s+="%.12e " % a
  print s
  evals_file.write("%s\n" % s)
  evals_file.flush()
  
  s = "rank %d returning from array_function:" % (world.rank)
  sum = 0
  for r in result:
    s += " %e " % r
    sum += r*r
  s += "   %e %e" % (sum, math.sqrt(sum))
  print s

  evals_file.write("%s\n" % s)
  evals_file.flush()

def report_bad_result(args, result, evals_file):
  from boost.mpi import world
  import math
  s = "rank %d out of bounds at x = " % (world.rank)
  for a in args:
    s+="%.12e " % a
  print s
  evals_file.write("%s\n" % s)
  evals_file.flush()
  
  s = "rank %d out of bounds returned :" % (world.rank)
  sum = 0
  for r in result:
    s += " %e " % r
    sum += r*r
  s += "   %e %e" % (sum, math.sqrt(sum))
  print s

  evals_file.write("%s\n" % s)
  evals_file.flush()


def simple_printout(eigs, pc, pressure, occs):
  import numpy as np
  dims = np.shape(eigs)
  print "---------------------------------------"
  print "eigs = [%d X %d]" % (dims[0], dims[1])
  for i in range(0,dims[0]):
    s = ""
    for j in range(0,dims[1]):
      s += "%f " % eigs[i][j]
    print s
  print "pc = ", pc
  print "occs = ", occs
  print "pressure = ", pressure
  print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  

def create_sys_dirs(systems, run_control):
  from os import  path
  import os
#  if (len(systems.systems) > 1):
# run dir for every system
  for s in systems.systems:
    sysdir = path.join(run_control.outdir, "%s" % (s.name))
    if (not path.isdir(sysdir)):
      os.mkdir(sysdir)

def create_directories(run_control, systems, job_comm, mycolor):
  from boost.mpi import world
  import os
  from os import  path
  basedir = run_control.outdir
  # world rank 0 creates ALL the directories, no conflict
  if (world.rank == 0):
    # creates path where VASP runs are stored.
    if (not path.isdir(basedir)):
      os.mkdir(basedir)
#    if run_control.nbjobs > 1:
    for color in range(0,run_control.nbjobs):
      jobdir = path.join(basedir, "job%i" % (color))
      run_control.outdir = jobdir
      if (not path.isdir(jobdir)):
        os.mkdir(jobdir)
      # further extension for ranks within a job
      if (job_comm.size > 1):
        for rank in range(0,job_comm.size):
          rankdir = path.join(jobdir, "rank%i" % (rank))
          if (not path.isdir(rankdir)):
            os.mkdir(rankdir)
          run_control.outdir = rankdir
          create_sys_dirs(systems, run_control)
      else:
        create_sys_dirs(systems, run_control)

# else:
#      create_sys_dirs(systems, run_control)
          

  # but each rank will USE certain directories
  run_control.outdir = basedir
  jobdir = basedir
#  if run_control.nbjobs > 1:
  run_control.outdir = path.join(run_control.outdir, "job%i" % (mycolor))
  jobdir = basedir
  # further extension for ranks within a job
  if (job_comm.size > 1):
    run_control.outdir = path.join(run_control.outdir, "rank%i" % (job_comm.rank))
  # and for different systems
#  if (len(systems.systems) > 1):      
  for s in systems.systems:
    s.outdir = path.join(run_control.outdir, "%s" % (s.name))

class RunControl():
  def __init__(self, run_input):
    self.outdir = run_input.outdir
    self.nbjobs = run_input.nbjobs
    self.optimizer = run_input.optimizer
  

def job_main(job_comm, run_input,  systems, vasp_comm):
  import random
  import numpy as np
  from scipy.optimize import leastsq, fmin as simplex

  from boost.mpi import world, broadcast

  use_test_objective = False

############ we are using some local functions, b/c they share data with the parent function ################
  def array_function(args, objective, sys_input, special_pc, evals_file):
    """ Returns an array that is (weighted) term by term fit to data.
    out of bounds arguments trigger penalty, avoids call to VASP"""
    #    result = check_bounds(args, sys_input, run_input, x0, ranges)
    # WARNING: this function now assumes that bounds have already been checked,
    # ie, that it is only called from array_function_multi
    if (not use_test_objective):
      eigs, pc, pressure, occs = objective(args)
    else:
      # the test objective is lame, just to see if code runs, not whether optimizer really works
      eigs, pc, pressure, occs = test_objective(args, sys_input)

    result = array_function_calcs(args, eigs, pc, pressure, occs, sys_input, run_input, special_pc, ignore_weights=False)
    report_result(args, result, "compound ", evals_file)
    return result
    
  def array_function_multi(args):
    # first check for bounds problems,
    ob = False
    for s in systems.systems:
      x = systems.mapx_to_system(args, s)
      sub_ranges = systems.mapx_to_system(ranges, s)
      sub_x0 = systems.mapx_to_system(x0, s)
      result = check_bounds(x, s.input, run_input, sub_x0, sub_ranges)
      if (result != None):
        report_bad_result(x, result, s.evals_file)
        ob = True
        # now result has penalized result for one system.  but that's not the right size
        # for returning.  just duplicate that badness til it's right length
        sz = systems.get_result_size(run_input)
        res = [result[0] for i in range(0,sz)]
      
    if (ob):
      result = res
    else:
      result = []
      # if no out of bounds, then do real calcs.
      for s in systems.systems:
        x = systems.mapx_to_system(args, s)
        wprint ("mapped ", args, " to ", x)
        res = array_function(x, s.objective, s.input, s.special_pc, s.evals_file)
        wprint ("array function returning: ", res)
        for r in res:
          result.append(r)
      wprint ("array function MULTI returning: ", result)
      report_result(args, result, "multi ", systems.evals_file)
    return result

  def array_function_multi2(args, fx):
    ## called from lmfit, tricky python->C->python interface, can't "return" a list
    result = array_function_multi(args)
    for i in range(0,len(result)):
      fx[i] = result[i]
################# end of local functions ############################  
  
  # if x0 provided, use it; otherwise randomizes starting point for each job_comm
  # first job uses x0 from specification of nlep params in input.py
  # if x0 provided, other jobs use it in order;
  # otherwise randomize starting point for each job_comm
  x0 = systems.getx()
  ranges = systems.get_ranges()
  print "ranges = ", ranges

  if (job_comm.rank == 0 and (run_input.random_start==None or run_input.random_start==False)):
    x0 = systems.getx()
    wprint("using default initial condition ", x0)
  else:
    x0 = np.array([x0[i] + random.uniform(-ranges[i], ranges[i]) for i in range(0,len(x0))], dtype="float64")
    wprint("setting random initial condition ", x0)
#  elif (not input.x0 == None  and world.rank-1 < len(input.x0)):
#    x0 = np.array(input.x0[world.rank-1])   ##THIS IS A BUG, if we specify x0 in input_generic.py
#  elif input.bounds != None:
#    x0 = np.array([random.uniform(*input.bounds) for u in x0], dtype="float64")
#  else:
#    x0 = np.zeros((x0.size), dtype="float64")
    #  objective.x = broadcast(job_comm, x0, 0)
    #  if world.rank == 0: print "x0: ", x0
    
  print "rank %d    x0 = " % world.rank, x0
  print "rank %d    ranges = " % world.rank, ranges

##########
  #### need fix.
  if (run_input.one_shot != None and run_input.one_shot == True):
    array_function_multi(x0)
##########
  
  if ((run_input.one_shot == None or not run_input.one_shot) and not run_input.do_scan):
    if run_input.optimizer == "leastsq": # least-square fit with scipy.optimize.leastsq
      # Somehow leastsq expects a function, but never a functor!!
      if (job_comm.rank == 0):
        # print some stuff:
        print "epsfcn = ", run_input.epsfcn
        print "xtol = ", run_input.xtol
        print "ftol = ", run_input.ftol
        print "factor = ", run_input.step_factor
        print "Calling scipy least squares fitter"
        x0, cov_x, infodict, mesg, ier\
            = leastsq( array_function_multi, x0, maxfev=run_input.max_functional_eval,\
                       full_output=1, xtol=run_input.xtol, ftol=run_input.ftol,\
                       factor = run_input.step_factor, epsfcn=run_input.epsfcn )
        
        # computes/print optimal results.
#        result = objective.final(x0)
        print "rank %d minimum value of sum of squares:" % world.rank, np.linalg.norm(infodict["fvec"])
        print "for: ", x0 
        print "after %i function calls." % (infodict["nfev"])
        print "with warning flag: ", ier, mesg 
    elif run_input.optimizer == "lmmin":  # least square via lmmin (parallel finite difference derivative)
      import liblmmin as lm
      npar = len(x0)
      ndat = systems.get_result_size(run_input)
      print "Calling lmmin fd-parallel least squares fitter with epsfcn = ", run_input.epsfcn
      x0 = np.array([float(xi) for xi in x0])
      lm.lmmin_py(job_comm, array_function_multi2, npar, x0, ndat, run_input.epsfcn)
      print "lmmin_py returned!"
    else:
      print "YOU DID NOT SPECIFY A FITTING METHOD"
      
  for s in systems.systems:
    s.evals_file.close()
  systems.evals_file.close()

def load_options():
    from optparse import OptionParser
    parser = OptionParser()    
    parser.add_option("-t", "--run-input", dest="run_input_name",  type="string", default="run_input.py",
                      help="name of input file")
    (options, args) = parser.parse_args()
    return (options, args)

def load_run_input(): 
  from pylada.vasp import read_input
  #  options, args = load_options()
  rin = "run_input.py"
  run_input = read_input("input_generic.py")
#  run_input.update(read_input(options.run_input_name))
  run_input.update(read_input(rin))
  return run_input
  
def main(system_params=None):
  from boost.mpi import world, broadcast
  from pylada.vasp.extract import Extract, ExtractGW
  from pylada.vasp.specie import U, nlep
  from pylada.vasp import Vasp
  import os
  from os import  path
  from nlep import Objective
  from pylada.vasp.nlep.postprocessing import find_best, load_test, prepare_analog_fit

  print "mpi rank %d of %d" % (world.rank, world.size)

  # Things which need to be put into dictionary of input script.
#  global_dict = { "Extract": Extract, "ExtractGW": ExtractGW, "Vasp": Vasp, "nlep": nlep, "U": U }

  # read file governing run.  other input files should be generic
  run_input = load_run_input()
  run_control = RunControl(run_input)
  
  load_from_analogs = run_input.load_from_analogs
  # create systems
  from systems import setup_systems
  systems = setup_systems(run_input, system_params)
      
  if (run_input.nbjobs == -1):
    run_control.nbjobs = world.size
  assert world.size >= run_control.nbjobs, "Too few procs granted, or too many jobs requested."
  job_comm = world
  vasp_comm = world
  color = world.rank % run_control.nbjobs
  job_comm = world.split(color)
  vasp_comm = job_comm
  if (job_comm.size > 1):
    vasp_comm = job_comm.split(job_comm.rank)
    
  create_directories(run_control,systems, job_comm, color)

  # barrier to be sure all necessary directories exist
  world.barrier()

  # creates objective functions from systems
  for s in systems.systems:
    print s.outdir
    s.objective = Objective( s.input.vasp, s.input.dft_in, s.input.gw_in,\
                             outdir=s.outdir, comm=vasp_comm, units=run_input.units )

    s.evals_file = file("%s/evals" % s.outdir, "w")
  systems.evals_file = file("%s/evals" % run_control.outdir, "w")
  # setup correspondence between vector "x" and list of species (ie at least one subpart shared by > 1 systems):
  systems.setup_species()

  # barrier to be sure all necessary directories exist
  world.barrier()
    
  # only serial vasp so far, but lmmin is a parallel fitter
  if (job_comm.rank != 0 and run_input.optimizer != "lmmin"):
    print "rank %d superfluous, returning; beware barrier" % world.rank
    return

  print "world rank %d    job %d     local rank %d    working dir %s" % (world.rank, color, job_comm.rank, run_control.outdir)

  job_main(job_comm, run_input, systems, vasp_comm)


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    from boost.mpi import world, abort
    import traceback
    from sys import stderr
    print >> stderr, "Final exception encountered by process: ", world.rank
    traceback.print_exc(file=stderr)
    abort(0)
