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
#  define PYLADA_UNITY math::iVector3d(1,1,1)
#  define PYLADA_ZERO math::iVector3d(0,0,0)
#  define PYLADA_SMALL math::iVector3d(0, 0, 0)
#  define PYLADA_TOL   1
#elif PYLADA_TEST_INCTYPE == 1
#  define PYLADA_ZERO math::iMatrix3d::Zero()
#  define PYLADA_SMALL math::iMatrix3d::Zero()
#  define PYLADA_TOL   1
#elif PYLADA_TEST_INCTYPE == 2
#  define PYLADA_UNITY math::rVector3d(1,1,1)
#  define PYLADA_ZERO math::rVector3d(0,0,0)
#  define PYLADA_SMALL math::rVector3d(0.5 * types::tolerance, 0, 0)
#  define PYLADA_TOL   types::tolerance
#elif PYLADA_TEST_INCTYPE == 3
#  define PYLADA_ZERO math::rMatrix3d::Zero()
#  define PYLADA_TOL   types::tolerance
#endif

using namespace std;
int main()
{
  using namespace Pylada;
  using namespace Pylada::math;
# if PYLADA_TEST_INCTYPE == 1
    iMatrix3d PYLADA_UNITY;
    PYLADA_UNITY << 1, 0, 0, 0, 1, 0, 0, 0, 1;
# elif PYLADA_TEST_INCTYPE == 3
    rMatrix3d PYLADA_UNITY, PYLADA_SMALL;
    PYLADA_UNITY << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    PYLADA_SMALL << 0.5 * types::tolerance, 0, 0, 0, 0.5 * types::tolerance, 0, 0.5 * types::tolerance, 0, 0;
# endif

  PYLADA_DOASSERT(is_identity(PYLADA_UNITY+PYLADA_SMALL), "unexpected unity.")
# if PYLADA_TEST_TYPE > 1
  PYLADA_DOASSERT(not is_identity(PYLADA_UNITY+10*PYLADA_SMALL, PYLADA_TOL), "unexpected unity.")
# endif
  PYLADA_DOASSERT(not is_identity(2*PYLADA_UNITY+PYLADA_SMALL, PYLADA_TOL), "unexpected unity.")
  PYLADA_DOASSERT(is_null(PYLADA_ZERO + PYLADA_SMALL), "unexpected null.")
  PYLADA_DOASSERT(not is_null(PYLADA_UNITY + PYLADA_SMALL, PYLADA_TOL), "unexpected null.")
  PYLADA_DOASSERT(is_integer(PYLADA_UNITY + PYLADA_SMALL), "unexpected integer.")
  PYLADA_DOASSERT(eq(PYLADA_UNITY + PYLADA_SMALL, PYLADA_UNITY), "unexpected eq.")
  PYLADA_DOASSERT(neq(2*PYLADA_UNITY, PYLADA_UNITY), "unexpected neq.")
# if PYLADA_TEST_TYPE > 1
    PYLADA_DOASSERT(not is_integer(PYLADA_UNITY + PYLADA_SMALL, PYLADA_TOL), "unexpected integer.")
# endif
# if PYLADA_TEST_TYPE == 3
    matrix << 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0;
    PYLADA_DOASSERT(are_periodic_images( rVector3d(0.5, 0.25, 0.5),
                                     rVector3d(0.5, 0.25, 0.5) + matrix * rVector3d(1, -10, 2),
                                     matrix.inverse() ), "NOT PERIODIC.\n" );
# endif
  return 0;
}
