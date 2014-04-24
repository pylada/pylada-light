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

#include<iostream>

#define PYLADA_DOASSERT(a,b) \
        { \
          if((not (a)))\
          { \
            std::cerr << __FILE__ << ", line: " << __LINE__ << "\n" << b; \
            throw 0;\
          }\
        }

#include "../math.h"

#if PYLADA_TEST_INCTYPE == 0
#  define PYLADA_FIRST 5
#  define PYLADA_SMALL 0
#  define PYLADA_TOL   1
#elif PYLADA_TEST_INCTYPE == 1
#  define PYLADA_FIRST 5e0
#  define PYLADA_SMALL   0.5 * types::tolerance
#  define PYLADA_TOL   10 * types::tolerance
#endif

using namespace std;
int main()
{
  using namespace Pylada;
  using namespace Pylada::math;

  PYLADA_DOASSERT(eq(PYLADA_FIRST, PYLADA_FIRST+PYLADA_SMALL), "unexpected eq result.\n")
  PYLADA_DOASSERT(not eq(PYLADA_FIRST, PYLADA_FIRST+1), "unexpected eq result.\n")
  PYLADA_DOASSERT(not lt(PYLADA_FIRST, PYLADA_FIRST+PYLADA_SMALL), "unexpected lt result.\n")
  PYLADA_DOASSERT(not lt(PYLADA_FIRST, PYLADA_FIRST+PYLADA_SMALL, PYLADA_TOL), "unexpected lt result.\n")
  PYLADA_DOASSERT(lt(PYLADA_FIRST, PYLADA_FIRST+5), "unexpected lt result.\n")
  PYLADA_DOASSERT(not gt(PYLADA_FIRST, PYLADA_FIRST-PYLADA_SMALL), "unexpected gt result.\n")
  PYLADA_DOASSERT(not gt(PYLADA_FIRST, PYLADA_FIRST-PYLADA_SMALL, PYLADA_TOL), "unexpected gt result.\n")
  PYLADA_DOASSERT(gt(PYLADA_FIRST+5, PYLADA_FIRST), "unexpected gt result.\n")
  PYLADA_DOASSERT(is_identity(1 + PYLADA_SMALL), "unexpected unity.")
  PYLADA_DOASSERT(not is_identity(PYLADA_FIRST), "unexpected unity.")
  PYLADA_DOASSERT(is_null(PYLADA_SMALL), "unexpected null.")
  PYLADA_DOASSERT(not is_null(PYLADA_FIRST), "unexpected null.")
  PYLADA_DOASSERT(is_integer(PYLADA_FIRST + PYLADA_SMALL), "unexpected integer.")
# if PYLADA_TEST_TYPE != 0
    PYLADA_DOASSERT(not is_integer(PYLADA_FIRST + PYLADA_SMALL, PYLADA_TOL), "unexpected integer.")
# endif
  return 0;
}
