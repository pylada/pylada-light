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

""" multifit NLEP fitting input script. """
from os.path import join, exists
from shutil import rmtree
import numpy as np
import os

from system_params import SystemParams

class System():
    def __init__(self):
        self.name = ""
        self.input = None
        self.objective = None
        self.special_pc = None
        self.outdir = None
        self.floating_vbm = False
    def __init__(self, name, input, outdir):
        self.name = name
        self.input = input
        self.objective = None
        self.special_pc = None
        self.floating_vbm = False
        self.outdir = outdir

def setup_one_system(cat, an, run_input, nlep_params, sys_params=None):
    from pylada.vasp import read_input
    from pylada.vasp.extract import Extract, ExtractGW
    from pylada.vasp.specie import U, nlep
    from pylada.vasp import Vasp
    import os
    from boost.mpi import world
    compound = "%s%s" % (cat, an)
    
    print "creating system for ", cat, an, " with nlep_params= ", nlep_params
    potcar_dir = run_input.potcar_dir
    outcar_data_dir = run_input.outcar_data_dir
    outdir = run_input.outdir
    if (sys_params == None):
        sys_params = SystemParams(cat, an, potcar_dir, outcar_data_dir, run_input.floating_vbm, nlep_params)
    else:
        print "using user system params for %s" % (compound)
    theSys = System(compound, sys_params, outdir)
    if (run_input.special_pc != None and compound in run_input.special_pc):
        print "overriding target partial charges for systems %s from run_input" % compound
        theSys.special_pc = run_input.special_pc[compound]
    if (run_input.floating_vbm):
        theSys.floating_vbm = True
    if (run_input.eigenvalue_weights != None and compound in run_input.eigenvalue_weights):
        print "overriding eigenvalue weights for systems %s from run_input" % compound
        theSys.eigenvalue_weights = run_input.eigenvalue_weights[compound]

    return theSys


class MultiSystem():
    def __init__(self, run_input, system_params = None):
        from pylada.vasp.nlep.postprocessing import load_run_input, find_best, load_test, prepare_analog_fit, get_analog_name
        from boost.mpi import world
        cations = run_input.cations
        anions = run_input.anions
        dont_fit = run_input.dont_fit
        self.cations = cations
        self.anions = anions
        self.objectives = None
        self.systems = []
        self.descriptor = str(cations) + str(anions)

        if (run_input.load_from_analogs):
            alog_system_names= []
            for cat in cations:
                for an in anions:
                    cmpd = get_analog_name (cat, an)
                    alog_system_names.append(cmpd)

            run_input=load_run_input()
            withranks = (run_input.optimizer == "lmmin")
            if (withranks):
#                rank = world.rank
                rank = 0
            else:
                rank = None
            
            job, runs = find_best(alog_system_names, run_input.nbjobs, withranks)

        for cat in cations:
            for an in anions:
                cmpd = "%s%s" % (cat, an)
                print "check: is ", cmpd, " in ", dont_fit
                if (dont_fit == None or cmpd not in dont_fit):
                    if (run_input.load_from_analogs):
                        analog_cmpd = get_analog_name(cat, an)
                        test, nlep = load_test(runs[analog_cmpd], job, rank, analog_cmpd, None)    
                        nlep_params = prepare_analog_fit(test, nlep)
                        print "nlep_params from analog fit: ", nlep_params
                    else:
                        nlep_params = None
                    # append these, don't require all
                    if (run_input.nlep_params != None):
                        if (nlep_params == None):
                            nlep_params = run_input.nlep_params
                        else:
                            for key in run_input.nlep_params:                                
                                nlep_params[key]  = run_input.nlep_params[key]                                            

                    if (system_params != None and cmpd in system_params):
                        this_sys_params = system_params[cmpd]
                    else:
                        this_sys_params = None
                    newsystem = setup_one_system(cat, an, run_input, nlep_params, this_sys_params)
                    self.systems.append(newsystem)
        self.result_size = self.get_result_size(run_input)

    def get_nlep_params_x(self):
        from nlep import getx_from_specie, set_nlep_fromx
        x = []
        for symbol, specie in self.species.items():
            x += getx_from_specie(specie)
        return x

    def getx(self):
        x = self.get_nlep_params_x()
        for s in self.systems:
            if (False and s.floating_vbm):   # the variable that stores the band_shift is at the end of the big multi-system x
                print "floating vbm"
                x.append(0)  # initial shift is zero
        print x        
        return np.array(x)

    def get_ranges(self):
        from nlep import get_range_from_specie
        x = []
        for symbol, specie in self.species.items():
            x += get_range_from_specie(specie)
        print x        
        return np.array(x)
    
    def setx(self, x):
        from nlep import getx_from_specie, set_nlep_fromx
        i = 0
        for symbol, specie in self.species.items():
            i = set_nlep_fromx(x, i, specie)
            
    def setup_species(self):
        """extract species from vasp objects to make one list for whole set of systems"""
        from nlep import getx_from_specie, set_nlep_fromx
        self.species = {}
        for s in self.systems:
            for symbol, specie in s.objective.vasp.species.items():
#                print "system symbol is", symbol
                self.species[symbol] = specie
        idx = 0
        self.species_dict = {}
        for symbol, specie in self.species.items():
#            print symbol, specie
            xspan = len(getx_from_specie(specie))
            spec = SpecRec(symbol, specie, idx, idx+xspan)
            idx += xspan
            self.species_dict[symbol] = spec

    def get_result_size(self, run_input):
        from mpopt import get_result_size
        size = 0
        for s in self.systems:
            size += get_result_size(s.input, run_input)
        return size
    
    def mapx_to_system(self, x, sys):
        import numpy as np
        xsys = []
        for symbol, specie in sys.objective.vasp.species.items():
            spec = self.species_dict[symbol]
            for e in x[spec.xstart:spec.xend]:
                xsys.append(e)
        if (False and sys.floating_vbm):  # the variable that stores the band_shift is at the end of the big multi-system x
            total_num_nlep_params = len(self.get_nlep_params_x())
            if (len(x) > total_num_nlep_params):
                for i in range(0,len(self.systems)):
                    if (self.systems[i] == sys):
                        print "found sys ", i, total_num_nlep_params, x
                        break
                xsys.append(x[total_num_nlep_params + i])
        return np.array(xsys)


class SpecRec():
    def __init__(self, symbol, specie, xstart, xend):
        self.symbol = symbol
        self.specie = specie
        self.xstart = xstart
        self.xend = xend


def setup_systems(run_input, system_params):
    systems = MultiSystem(run_input, system_params)
    return systems
            
if __name__ == '__main__':
    systems = setup_systems()
    systems.setup_species()
    systems.getx()

