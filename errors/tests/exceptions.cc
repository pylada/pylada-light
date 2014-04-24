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

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <root_exceptions.h>
#include "../exceptions.h"

namespace bp = boost::python;

void nothrow () {};
void dothrow_nomessage() {
  BOOST_THROW_EXCEPTION(::Pylada::error::PYLADA_TYPE()); 
}
void dothrow_message()
{
  std::string message = "This is a message.";
  BOOST_THROW_EXCEPTION(::Pylada::error::PYLADA_TYPE() << Pylada::error::string(message)); 
}
void dopythrow_message()
{
  PyErr_SetString(::Pylada::error::get_error(PYLADA_TYPENAME).ptr(), "This is another message.");
  std::string message = "This is a message.";
  BOOST_THROW_EXCEPTION(::Pylada::error::PYLADA_TYPE() << Pylada::error::string(message)); 
}

//void BOOST_THROW_EXCEPTION( error::root exc) {
//  throw exc.what();
//}

BOOST_PYTHON_MODULE(PYLADA_MODULE)
{
  Pylada::error::bp_register();
  bp::def("nothrow", &nothrow);
  bp::def("dothrow_nomessage", &dothrow_nomessage);
  bp::def("dothrow_message", &dothrow_message);
  bp::def("dopythrow_message", &dopythrow_message);
}
