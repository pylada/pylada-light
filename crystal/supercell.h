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

#if PYLADA_CRYSTAL_MODULE != 1
  template<class T_DERIVED>
    Structure supercell(Structure const &_lattice, Eigen::DenseBase<T_DERIVED> const &_supercell)
    {
      if(_lattice.size() == 0) 
      {
        PYLADA_PYTHROW(ValueError, "Lattice is empty.");
        return Structure();
      }
      //namespace bt = boost::tuples;
      Structure result = _lattice.copy(); 
      if(not result) { PYLADA_PYTHROW(ValueError, "Could not deepcopy the lattice.");}
      result.clear();
      result->cell = _supercell;
      if(_lattice.hasattr("name") and PyString_Check(_lattice.pyattr("name").borrowed()) )
      {
        char *const attr = PyString_AS_STRING(_lattice.pyattr("name").borrowed());
        if(std::string(attr) != "")
        {
          std::string const name = "supercell of " + std::string(attr);
          PyObject* pyname = PyString_FromString(name.c_str());
          if(not pyname) { PYLADA_PYTHROW(internal, "Could not create string."); }
          result.pyattr("name", pyname);
          Py_DECREF(pyname);
          if(PyErr_Occurred()) BOOST_THROW_EXCEPTION(error::internal());
        }
      }
      HFTransform transform( _lattice.cell(), result.cell());
      if(not transform) return Structure();;
    
      const math::rMatrix3d factor(transform.transform().inverse());
      math::rMatrix3d inv_cell( result.cell().inverse() ); 
      result.reserve(transform.size()*_lattice.size());
      Structure::const_iterator const i_site_begin = _lattice.begin();
      Structure::const_iterator const i_site_end = _lattice.end();
      
      for( math::iVector3d::Scalar i(0); i < transform.quotient()(0); ++i )
        for( math::iVector3d::Scalar j(0); j < transform.quotient()(1); ++j )
          for( math::iVector3d::Scalar k(0); k < transform.quotient()(2); ++k )
          {
            // in cartesian.
            const math::rVector3d vec( factor * math::rVector3d(i,j,k) );
          
            // adds all lattice sites.
            long l(0);
            for(Structure::const_iterator i_site(i_site_begin); i_site != i_site_end; ++i_site, ++l)
            {
              Atom atom = i_site->copy();
              if(not atom) PYLADA_PYTHROW(ValueError, "Could not deepcopy atom.");
              atom->pos = into_cell(vec+i_site->pos(), result->cell, inv_cell);
              python::Object site = PyInt_FromLong(l);
              if(not atom) PYLADA_PYTHROW(internal, "Could not create python integer.");
              if(not atom.pyattr("site", site) ) 
                PYLADA_PYTHROW(internal, "Could not set site index attribute.");
              result.push_back(atom);
            }
          }
    
      return result;
    }
#endif
