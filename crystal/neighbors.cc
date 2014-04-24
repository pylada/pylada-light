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

//! Inserts a neighbor in a list.
bool insert_neighbor_impl_( python::Object const &_list, 
                            Atom const &_ref,
                            math::rVector3d const &_pos,
                            types::t_real _distance,
                            Py_ssize_t _i )
{
  PyObject *pydist = PyFloat_FromDouble(_distance);
  if(not pydist) return false;
  PyObject* pypos = python::numpy::wrap_to_numpy(_pos);
  if(not pypos) {Py_DECREF(pydist); return false; }
  PyObject *tuple = PyTuple_Pack(3, _ref.borrowed(), pypos, pydist);
  Py_DECREF(pypos); Py_DECREF(pydist);
  if(not tuple) return false;
  bool result = PyList_Insert(_list.borrowed(), _i, tuple) == 0;
  Py_DECREF(tuple);
  return result;
}
//! Appends a neighbor in a list.
bool append_neighbor_impl_( python::Object const &_list, 
                            Atom const &_ref,
                            math::rVector3d const &_pos,
                            types::t_real _distance )
{
  PyObject *pydist = PyFloat_FromDouble(_distance);
  if(not pydist) return false;
  PyObject* pypos = python::numpy::wrap_to_numpy(_pos);
  if(not pypos) {Py_DECREF(pydist); return false; }
  PyObject *tuple = PyTuple_Pack(3, _ref.borrowed(), pypos, pydist);
  Py_DECREF(pypos); Py_DECREF(pydist);
  if(not tuple) return false;
  bool result = PyList_Append(_list.borrowed(), tuple) == 0;
  Py_DECREF(tuple);
  return result;
}
      
#ifdef PYLADA_DISTANCE
#  error PYLADA_DISTANCE already defined.
#endif
 //! \macro PYLADA_DISTANCE returns distance attribute from nth item in list.
#define PYLADA_DISTANCE(list, n)  \
   PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(PyList_GET_ITEM(list, n), 2))
     
      
PyObject* neighbors( crystal::Structure const &_structure, Py_ssize_t _nmax, 
                     math::rVector3d const &_center,
                     types::t_real _tolerance )
{
  const types::t_int N( _structure.size() );
  
  math::rMatrix3d const cell = math::gruber(_structure.cell(), 3e0*_tolerance);
  math::rMatrix3d const inv_cell( !cell );

  types::t_real const volume(_structure.volume());
  types::t_int list_max_size(_nmax+2);

  python::Object result = PyList_New(0);
  if(not result) return NULL;
  
  retry: 

  types::t_int size(0);
  // Finds out how far to look.
  math::rVector3d const a0(cell.col(0));
  math::rVector3d const a1(cell.col(1));
  math::rVector3d const a2(cell.col(2));
  types::t_real const max_norm
    = std::max( a0.norm(), std::max(a1.norm(), a2.norm()) );
  types::t_real const r
  ( 
    std::pow
    (
      std::max(1e0, types::t_real(list_max_size) / types::t_real(N)),
      0.3333333333333
    )
  );
  types::t_int n0( std::max(1.0, std::ceil(r*max_norm*a1.cross(a2).norm()/volume)) );
  types::t_int n1( std::max(1.0, std::ceil(r*max_norm*a2.cross(a0).norm()/volume)) );
  types::t_int n2( std::max(1.0, std::ceil(r*max_norm*a0.cross(a1).norm()/volume)) );
  while( n0 * n1 * n2 * 8 * N < list_max_size ) { ++n0; ++n1; ++n2; }


  types::t_real max_distance( 1.2 * std::pow(volume/types::t_real(N), 2e0/3e0)
                                  * types::t_real(list_max_size * list_max_size) );
  Structure::t_Atoms::const_iterator i_atom = _structure.begin();
  Structure::t_Atoms::const_iterator i_atom_end = _structure.end();
  for(; i_atom != i_atom_end; ++i_atom) 
  {
    math::rVector3d const start = into_voronoi(i_atom->pos()-_center, cell, inv_cell);
    if(start.squaredNorm() > max_distance) continue;
    for( types::t_int x(-n0); x <= n0; ++x )
      for( types::t_int y(-n1); y <= n1; ++y )
        for( types::t_int z(-n2); z <= n2; ++z )
        {
           math::rVector3d const pos = start + cell * math::rVector3d(x,y,z);
           types::t_real const distance = pos.norm();
           if(math::is_null(distance, _tolerance)) continue;
  
           Py_ssize_t i_found(0);
           Py_ssize_t const N(PyList_GET_SIZE(result.borrowed()));
           for(; i_found < N; ++i_found )
           {
             double const dist = PYLADA_DISTANCE(result.borrowed(), i_found);
             if(distance < dist) break;
           }
           if( i_found != N)
           {
             insert_neighbor_impl_(result, *i_atom, pos, distance, i_found);
             if( size < list_max_size ) ++size;
             else if(PySequence_DelItem(result.borrowed(), N) == -1) return NULL;
           }
           else if( size < list_max_size-1 )
           {
             ++size;
             append_neighbor_impl_(result, *i_atom, pos, distance);
           }
        }
  } // loop over atoms.
  
  // Removes atoms beyond nth position which are not at same distance as nth position.
  // Also sets last item to last of same size as nmax.
  Py_ssize_t const rN(PyList_GET_SIZE(result.borrowed()));
  if(rN <= _nmax)
  {
    PYLADA_PYERROR(InternalError, "Supercell too small.");
    return NULL;
  }
  types::t_real lastdist = PYLADA_DISTANCE(result.borrowed(), _nmax-1);
  Py_ssize_t i(_nmax);
  for(; i < rN; ++i) 
    if(math::gt(PYLADA_DISTANCE(result.borrowed(), i), lastdist, _tolerance)) break;
  if(i == rN)
  {
    result.reset(PyList_New(0));
    if(not result) return NULL;
    list_max_size += 20;
    goto retry;
  }
  for(Py_ssize_t j(rN-1); j >= i; --j) PySequence_DelItem(result.borrowed(), j);
  return result.release();
}
#undef PYLADA_DISTANCE
