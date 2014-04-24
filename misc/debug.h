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

#ifndef _OPT_DEBUG_H_
#define _OPT_DEBUG_H_
#include <sstream>
#include <stdexcept>
#include <boost/foreach.hpp>
#include <iostream>
#include <boost/throw_exception.hpp>

#if defined(foreach)
#  warning "foreach macro already exists"
#endif
#define foreach BOOST_FOREACH

#define PYLADA_SPOT_ERROR __FILE__ << ", line: " << __LINE__ << "\n" 
#define PYLADA_CATCHCODE(code, error)\
        catch(...)\
        {\
          code;\
          std::cerr << PYLADA_SPOT_ERROR << error; \
          throw 0; \
        }
#define PYLADA_THROW_ERROR(error) \
        {\
          std::cerr << PYLADA_SPOT_ERROR << error ;\
          throw 0; \
        }
#define PYLADA_TRY_CODE(code,error) try { code } \
        catch ( std::exception &_e )\
        PYLADA_THROW_ERROR( error << _e.what() )
#define PYLADA_NASSERT_CATCHCODE(condition, code, error) \
          if( condition ) { code; PYLADA_THROW_ERROR( error ) }
#define PYLADA_DO_NASSERT( condition, error ) \
          PYLADA_NASSERT_CATCHCODE( condition, ,error )
#define PYLADA_TRY_ASSERT(condition, error) \
          PYLADA_TRY_CODE( PYLADA_DO_NASSERT( condition, "" ), error )
#define PYLADA_ASSERT(a,b) PYLADA_NASSERT( (not (a) ), b)
#define PYLADA_DOASSERT(a,b) \
        { \
          if((not (a)))\
          { \
            std::cerr << __FILE__ << ", line: " << __LINE__ << "\n" << b; \
            throw 0;\
          }\
        }
#define PYLADA_TRY_BEGIN try {
#define PYLADA_TRY_END( code, error ) } PYLADA_CATCHCODE( code, error ) 
#define PYLADA_COMMA ,
#define PYLADA_BEGINGROUP {
#define PYLADA_ENDGROUP }

#ifdef PYLADA_DEBUG
# include <boost/throw_exception.hpp>
# define PYLADA_BASSERT(condition, error) if(not (condition)) BOOST_THROW_EXCEPTION(error)
# define PYLADA_NASSERT(condition, error ) PYLADA_DO_NASSERT(condition, error)
# define PYLADA_DEBUG_TRY_BEGIN try {
# define PYLADA_DEBUG_TRY_END( code, error ) } PYLADA_CATCHCODE( code, error ) 
#else
# define PYLADA_BASSERT(condition, error) 
# define PYLADA_NASSERT( condition, error ) 
# define PYLADA_DEBUG_TRY_BEGIN 
# define PYLADA_DEBUG_TRY_END( code, error ) 
#endif

#endif
