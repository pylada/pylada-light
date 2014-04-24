/******************************
   This file is part of PyLaDa.

   Copyright (C) 2013 National Renewable Energy Lab
  
   PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
   large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
   crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
   is able to organise and launch computational jobs on PBS and SLURM.
  
   PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
   Public License as published by the Free Software Foundation, either version 3 of the License, or (at
   your option) any later version.
  
   PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.
  
   You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
   <http://www.gnu.org/licenses/>.
******************************/

#include "PyladaConfig.h"
#include <iostream>
#include <string>
#include <vector>
#include <set>

#include <math/misc.h>
#include "../structure.h"

#define PYLADA_DOASSERT(a,b) \
        { \
          if((not (a)))\
          { \
            std::cerr << __FILE__ << ", line: " << __LINE__ << "\n" << b; \
            throw 0;\
          }\
        }
#if PYLADA_INCREMENT == 0
#  define PYLADA_INIT 
#  define PYLADA_DOASSERT_TYPE0 PYLADA_DOASSERT(c[0]->type == "Si", "Wrong first type.\n"); 
#  define PYLADA_DOASSERT_TYPE1 PYLADA_DOASSERT(c[1]->type == "Ge", "Wrong first type.\n"); 
#elif PYLADA_INCREMENT == 1
#  define PYLADA_INIT , "Si"
#  define PYLADA_DOASSERT_TYPE0 \
     PYLADA_DOASSERT(c[0]->type.size() == 1, "Wrong number of types.\n");\
     PYLADA_DOASSERT(c[0]->type[0] == "Si", "Wrong first type.\n");
#  define PYLADA_DOASSERT_TYPE1 \
     PYLADA_DOASSERT(c[1]->type.size() == 2, "Wrong number of types.\n");\
     PYLADA_DOASSERT(c[1]->type[0] == "Ge", "Wrong first type.\n");\
     PYLADA_DOASSERT(c[1]->type[1] == "Si", "Wrong first type.\n");
#elif PYLADA_INCREMENT == 2
#  define PYLADA_INIT , "Si"
#  define PYLADA_DOASSERT_TYPE0 \
    { PYLADA_TYPE cmp; cmp.insert("Si"); \
      PYLADA_DOASSERT(c[0]->type.size() == 1, "Wrong number of types.\n");\
      PYLADA_DOASSERT(c[0]->type == cmp, "Wrong first type.\n"); }
#  define PYLADA_DOASSERT_TYPE1 \
    { PYLADA_TYPE cmp; cmp.insert("Si"); cmp.insert("Ge");                \
      PYLADA_DOASSERT(c[1]->type.size() == 2, "Wrong number of types.\n");\
      PYLADA_DOASSERT(c[1]->type == cmp, "Wrong first type.\n"); }
#endif
using namespace std;
int main()
{
  using namespace Pylada;
  using namespace Pylada::crystal;
  using namespace Pylada::math;
  Structure< PYLADA_TYPE > lattice;
  lattice.set_cell(0,0.5,0.5)
                  (0.5,0,0.5)
                  (0.5,0.5,0);
  lattice.add_atom(0,0,0, "Si")
                  (0.25,0.25,0.25, "Ge" PYLADA_INIT);
  lattice[0]->freeze = AtomFreezeMixin::frozen::X;
  lattice[1]->freeze = AtomFreezeMixin::frozen::Y;
  lattice[0]->site = 0;
  lattice[1]->site = 1;
  lattice->energy = -1;
  lattice->weight = -1;
  lattice->scale = 0.5;
  lattice->freeze = frozenstr::XX;
  lattice->name = "hello";

  Structure< PYLADA_TYPE > c = lattice.copy();
  PYLADA_DOASSERT(is_null(c->cell - lattice->cell), "Different cells.\n");
  PYLADA_DOASSERT(is_null(c->energy - lattice->energy), "Different energies.\n");
  PYLADA_DOASSERT(is_null(c->weight - lattice->weight), "Different weights.\n");
  PYLADA_DOASSERT(is_null(c->scale - lattice->scale), "Different scale.\n");
  PYLADA_DOASSERT(c->name == lattice->name, "Different names.\n");
  PYLADA_DOASSERT(c->freeze == lattice->freeze, "Different frozen dofs.\n");
  PYLADA_DOASSERT(c.size() == lattice.size(), "Different sizes.\n");
  PYLADA_DOASSERT(is_null(c[0]->pos - lattice[0]->pos), "Different atomic position 0.\n");
  PYLADA_DOASSERT(is_null(c[0]->pos - lattice[0]->pos), "Different atomic position 0.\n");
  PYLADA_DOASSERT(c[0]->freeze == lattice[0]->freeze, "Different frozen dofs at site 0.\n");
  PYLADA_DOASSERT(c[1]->freeze == lattice[1]->freeze, "Different frozen dofs at site 1.\n");
  PYLADA_DOASSERT(c[0]->site == lattice[0]->site, "Different site index at site 0.\n");
  PYLADA_DOASSERT(c[1]->site == lattice[1]->site, "Different site index at site 0.\n");
  PYLADA_DOASSERT_TYPE0;
  PYLADA_DOASSERT_TYPE1;

  return 0;
}
