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

#ifndef PYLADA_ROOTEXCEPTIONS_H
#define PYLADA_ROOTEXCEPTIONS_H

#include "PyladaConfig.h"

#include <string>
#include <stdexcept>

#include <Python.h>

//PG -- basically commenting this out, b/c boost already defines it
#ifndef BOOST_THROW_EXCEPTION
#define BOOST_THROW_EXCEPTION( x) { throw(x); }
#endif

namespace Pylada
{
  namespace error
  {
    //! Root exception for all pylada exceptions.
    struct root: virtual std::exception {};

    //! Root of input errors.
    struct input: virtual root {};

    //! \brief Root of internal error.
    //! \brief These should be programmer errors, rather than something the
    //!        users would see.
    struct internal: virtual root {};
    //! out-of-range error.
    struct out_of_range: virtual root {};
    //! \brief end-of-iteration error.
    //! \details Should be used to avoid infinit loops. 
    struct infinite_loop: virtual root {};

    //! Convenience error info type to capture strings.
    // typedef boost::error_info<struct string_info,std::string> string;
    //! \brief Convenience error infor type to capture python objects.
    //! \details No increffing or decreffing. A python exception should already
    //!          exist.
    typedef void * pyobject;

  }
}

#endif
