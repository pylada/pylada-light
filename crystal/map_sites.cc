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

bool map_sites( Structure const &_mapper, Structure &_mappee,
                python::Object _withocc, types::t_real _tolerance )
{
  if(_mapper.size() == 0) 
  {
    PYLADA_PYERROR(ValueError, "Empty mapper structure.");
    return false;
  }
  if(_mappee.size() == 0) 
  {
    PYLADA_PYERROR(ValueError, "Empty mappee structure.");
    return false;
  }

  math::rMatrix3d const cell = math::gruber(_mapper.cell());
  math::rMatrix3d const invcell = cell.inverse();
  bool withocc( PyCallable_Check(_withocc.borrowed()) );
  
  // check that mappee_ is a supercell of mapper_.
  types::t_real const mappee_scale 
    = python::get_quantity(_mappee->scale, _mapper->scale);
  types::t_real const mapper_scale = python::get_quantity(_mapper->scale);
  types::t_real const ratio = mappee_scale / mapper_scale;
  types::t_real tolerance = _tolerance / mapper_scale;
  math::rMatrix3d const intcell_ = invcell * _mappee.cell() * ratio;
  if(not math::is_integer(intcell_, _tolerance))
  {
    PYLADA_PYERROR(ValueError, "Mappee not a supercell of mapper.");
    return false;
  }

  // Copy mapper sites to a vector, making sure positiosn are in cell.
  std::vector<math::rVector3d> sites; 
  Structure::const_iterator i_mapper_site = _mapper.begin();
  Structure::const_iterator const i_mapper_site_end = _mapper.end();
  for(; i_mapper_site != i_mapper_site_end; ++i_mapper_site)
    sites.push_back(into_cell(i_mapper_site->pos(), cell, invcell));

  // loop over atoms in mappee and assign sites.
  bool allmapped = true;
  Structure::iterator i_atom = _mappee.begin();
  Structure::iterator const i_atom_end = _mappee.end();
  std::vector<math::rVector3d>::const_iterator const i_site_end = sites.end();
  for(; i_atom != i_atom_end; ++i_atom)
  {
    // loop over lattice sites, find two first neighbors.
    types::t_int fneigh_index = -1;
    types::t_int sneigh_index = -1;
    types::t_real fneigh_dist = -1;
    types::t_real sneigh_dist = -1;
    std::vector<math::rVector3d>::const_iterator i_site = sites.begin();
    for(size_t i(0); i_site != i_site_end; ++i_site, ++i)
    {
      types::t_real const norm 
        = math::absnorm(into_voronoi(ratio*i_atom->pos()-(*i_site), cell, invcell));
      if(fneigh_dist > norm or fneigh_index == -1) 
      {
        sneigh_dist = fneigh_dist;
        sneigh_index = fneigh_index;
        fneigh_dist = norm;
        fneigh_index = i;
      }
      else if(sneigh_dist > norm or sneigh_index == -1)
      {
        sneigh_dist = norm;
        sneigh_index = i;
      }
    }
    if( math::eq(fneigh_dist, sneigh_dist, tolerance) and sneigh_index != -1)
    {
      PYLADA_PYERROR(ValueError, "Found two atoms at the same site.");
      return false;
    }
    if(fneigh_dist > tolerance) 
    {
      fneigh_index = -1;
      allmapped = false; 
    }
    else if(withocc) 
    {
      python::Object result
        = PyObject_CallFunctionObjArgs( _withocc.borrowed(),
                                        _mapper[fneigh_index].borrowed(),
                                        i_atom->borrowed(), NULL );
      if(not result) BOOST_THROW_EXCEPTION(error::internal());
      if(PyBool_Check(result.borrowed()))
      {
        if(result.borrowed() == Py_False) fneigh_index = -1;
      }
      else if(PyLong_Check(result.borrowed()) or PyInt_Check(result.borrowed()))
      {
        int i = PyInt_AS_LONG(result.borrowed());
        if(i == -1 and PyErr_Occurred() != NULL) BOOST_THROW_EXCEPTION(error::internal());
        if(i == 0) fneigh_index = -1;
      }
      else
      {
        PYLADA_PYERROR(ValueError, "Callable is expected to return True or False");
        return false;
      }
    }
    if(fneigh_index == -1)
    {
      i_atom->pyattr("site", Py_None);
      if(PyErr_Occurred() != NULL) BOOST_THROW_EXCEPTION(error::internal());
    }
    else
    { 
      python::Object pyint = PyLong_FromLong(fneigh_index);
      if(not pyint)
      { 
        PyErr_Clear(); 
        PYLADA_PYERROR(internal, "Could not create python integer."); 
        return false;
      }
      i_atom->pyattr("site", pyint.borrowed());
    }
  }
  return allmapped;
}
