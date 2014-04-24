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

#! /usr/bin/env python
#from decorator import decorator


#@decorator
#def count_calls(method, *args, **kwargs):
#  """ Adds call counting to a method. """
#  result = method(*args, **kwargs)
#  args[0]._nbcalls += 1
#  return result

def getx_from_specie(specie):
  result = []
  for nlep_params in specie.U:
#    print "nlep_param = ", nlep_params
    if nlep_params["func"] == "nlep":
      if nlep_params["fitU"]: 
        result.append(nlep_params["U"]) 
    elif nlep_params["func"] == "enlep":
      if (nlep_params["fitU0"]):
        result.append(nlep_params["U0"]) # first energy
      if (nlep_params["fitU1"]):  
        result.append(nlep_params["U1"]) # second energy
  return result

def get_range_from_specie(specie):
  result = []
  for nlep_params in specie.U:
    #        print "nlep_param = ", nlep_params
    if nlep_params["func"] == "nlep":
      if nlep_params["fitU"]: 
        result.append(nlep_params["U_range"]) 
    elif nlep_params["func"] == "enlep":
      if (nlep_params["fitU0"]):
        result.append(nlep_params["U0_range"]) # first energy range
      if (nlep_params["fitU1"]):   
        result.append(nlep_params["U1_range"]) # second energy range
  return result


def set_nlep_fromx(args, i, specie):
  for nlep_params in specie.U:
    if nlep_params["func"] == "nlep": 
      if nlep_params["fitU0"]: 
        assert args.shape[0] > i, RuntimeError("%i > %i\n" % (args.shape[0], i))
        nlep_params["U"] = args[i] 
        i += 1
    elif nlep_params["func"] == "enlep": 
      if nlep_params["fitU0"]: 
        assert args.shape[0] > i, RuntimeError("%i > %i\n" % (args.shape[0], i))
        nlep_params["U0"] = args[i]   # first energy
        i += 1
      if nlep_params["fitU1"]: 
        assert args.shape[0] > i, RuntimeError("%i > %i\n" % (args.shape[0], i))
        nlep_params["U1"] = args[i] # second energy
        i += 1
  return i
      
class Objective(object): 
  """ Objective function to optimize. 

      The vasp object is the one to make actual VASP calls and should be set up
      prior to minimization.
  """
  def __init__(self, vasp, dft, gw, outdir="nlep_fit", comm = None, units = None):
    from os import makedirs, getcwd
    from os.path import exists
    from shutil import rmtree
    from boost.mpi import world
    from pylada.crystal import Structure

    self.gw = gw
    self.dft = dft
    self.gw.comm = comm
    self.dft.comm = comm
    # since comm has changed, makes sure there are no issues with caching
    self.gw.uncache()
    self.dft.uncache()

    self.vasp = vasp
    self.system = Structure(dft.structure)
    self._nbcalls = 0
    self.outdir = outdir
    self.comm = comm if comm != None else world
    self.units = units if units != None else 1e0

    self.use_syscall = True

    if self.comm.rank == 0 and exists(self.outdir):
      rmtree(self.outdir)
      makedirs(self.outdir)
    self.comm.barrier()

  def _get_x0(self):
    """ Returns vector of parameters from L{vasp} attribute. """
    from numpy import array
    result = []
    for symbol, specie in self.vasp.species.items():
      result += getx_from_specie(specie)
    return array(result, dtype="float64") * self.units

  def _set_x0(self, args):
    """ Sets L{vasp} attribute from input vector. """
    from numpy import array, multiply, sum
    i = 0
    args = args.copy() / self.units
    for symbol, specie in self.vasp.species.items():
      i = set_nlep_fromx(args, i, specie)

  x = property(_get_x0, _set_x0)
  """ Vector of parameters. """

  def syscall_vasp(self, this_outdir):
    import os
    from pylada.vasp import Extract
    cwd = os.getcwd()
    os.chdir(this_outdir)
#    print "NOW calling vasp from:   ", os.getcwd()
    os.system("vasp > stdout")

    out = Extract(outcar="")
    os.chdir(cwd)
    return out

  #@count_calls
  def __call__(self, args):
    import os
    from os.path import join
    from boost.mpi import world
    from numpy import array
    from pylada.opt.changedir import Changedir
    from pylada.vasp import files
    from pylada.vasp.extract import Extract
    # transfers parameters to vasp object
    self.x = args
    # performs calculation in new directory
#    this_outdir = join(self.outdir, str(self._nbcalls)),
    this_outdir = "%s/%d" % (self.outdir, self._nbcalls)
    this_outdir = os.path.abspath(this_outdir)
    if (not os.path.isdir(this_outdir)):
      os.mkdir(this_outdir)

    print "rank %d calling vasp in dir %s"  %(world.rank, this_outdir)

    if self.use_syscall:
      out = self.vasp\
            (self.system,
             outdir = this_outdir,
             comm = self.comm,
             repat = files.minimal + files.input,
             norun=True)
      out = self.syscall_vasp(this_outdir)
    else:
      out = self.vasp\
            (self.system,
             outdir = this_outdir,
             comm = self.comm,
             repat = files.minimal + files.input)

#    assert out.success,\
#           RuntimeError\
#           (
#             "VASP calculation in %s_%i did not complete."\
#             % (self.outdir, self._nbcalls)
#           )
    self._nbcalls += 1
    # return raw values for subsequent processing
    eigs = out.eigenvalues 
    pc = out.partial_charges
    pressure = out.pressure 
    occs = out.occupations

    return eigs, pc, pressure, occs

      
