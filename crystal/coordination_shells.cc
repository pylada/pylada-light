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

struct DataPoint
{
  PyObject* atom;
  math::rVector3d trans;
  types::t_real distance;
};

struct CmpDataPoints
{
  types::t_real tolerance;
  CmpDataPoints(types::t_real const &_tol) : tolerance(_tol) {}
  CmpDataPoints(CmpDataPoints const &_c) : tolerance(_c.tolerance) {}
  bool operator()(DataPoint const &_a, DataPoint const &_b) const
    { return math::lt(_a.distance, _b.distance, tolerance); }
};

PyObject* coordination_shells( crystal::Structure const &_structure, Py_ssize_t _nshells, 
                               math::rVector3d const &_center,
                               types::t_real _tolerance, Py_ssize_t _natoms )
{
  // first, computes and sorts nth neighbors.
  const types::t_int N( _structure.size() );
  math::rMatrix3d const cell = math::gruber(_structure.cell(), 3e0*_tolerance);
  math::rMatrix3d const inv_cell( !cell );
  types::t_real const volume(_structure.volume());

  // Tries to figure out how far to look in number of atoms.
  if(_natoms == 0)
  {
    // adds number of atoms for fcc.
    _natoms = 12;
    if(_nshells > 1) _natoms += 6;
    if(_nshells > 2) _natoms += 24;
    if(_nshells > 3) _natoms += 12;
    if(_nshells > 4) _natoms += 24;
    if(_nshells > 5) _natoms += 8;
    if(_nshells > 6) _natoms += 48;
    if(_nshells > 7) _natoms += 6;
    if(_nshells > 8) _natoms += 32;
    if(_nshells > 9) _natoms += 24*(_nshells-9);
    // security buffer.
    if(_natoms < 12) _natoms += 6;
    else _natoms += _natoms >> 4;
  }
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
      std::max(1e0, types::t_real(_natoms) / types::t_real(N)),
      0.3333333333333
    )
  );
  types::t_int n0( std::max(1.0, std::ceil(r*max_norm*a1.cross(a2).norm()/volume)) );
  types::t_int n1( std::max(1.0, std::ceil(r*max_norm*a2.cross(a0).norm()/volume)) );
  types::t_int n2( std::max(1.0, std::ceil(r*max_norm*a0.cross(a1).norm()/volume)) );
  while( n0 * n1 * n2 * 8 * N < _natoms ) { ++n0; ++n1; ++n2; }

  typedef std::vector< DataPoint > t_DataPoints;
  t_DataPoints datapoints;
  datapoints.reserve(n0*n1*n2*8*N);
  types::t_real max_distance( 1.2 * std::pow(volume/types::t_real(_natoms), 2e0/3e0)
                                  * types::t_real(_natoms * _natoms) );
  Structure::const_iterator i_atom = _structure.begin();
  Structure::const_iterator i_atom_end = _structure.end();
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
          DataPoint const point = {i_atom->borrowed(), pos, distance};
          datapoints.push_back(point);
        }
  }
  std::partial_sort
  ( 
    datapoints.begin(), datapoints.begin() + _natoms, datapoints.end(),
    CmpDataPoints(_tolerance) 
  );


  //! creates list of results.
  python::Object result = PyList_New(0);
  if(not result) return NULL;
  t_DataPoints::const_iterator i_point = datapoints.begin();
  t_DataPoints::const_iterator i_point_end = datapoints.end();
  types::t_real current_norm = i_point->distance;
  while(math::is_null(current_norm, _tolerance)) 
  {
    ++i_point;
    if(i_point == i_point_end)
    {
      PYLADA_PYERROR(InternalError, "Could not find any point at non-zero distance from origin.");
      return NULL;
    }
    current_norm = i_point->distance;
  }
  
  for(Py_ssize_t nshells(0); nshells < _nshells; ++nshells)
  {
    python::Object current_list = PyList_New(0);
    if(not current_list) return NULL;
    if(PyList_Append(result.borrowed(), current_list.borrowed()) != 0) return NULL; 
    for(; i_point != i_point_end; ++i_point)
    {
      // loop until norm changes.
      if( math::lt(current_norm, i_point->distance, _tolerance) )
        { current_norm = i_point->distance; break; }
      // adds point to result.
      PyObject* pydist = PyFloat_FromDouble(i_point->distance);
      if(not pydist) return NULL;
      PyObject* pypos = python::numpy::wrap_to_numpy(i_point->trans);
      if(not pypos) { Py_DECREF(pydist); return NULL; }
      PyObject *tuple = PyTuple_Pack(3, i_point->atom, pypos, pydist);
      Py_DECREF(pypos); Py_DECREF(pydist);
      if(not tuple) return NULL;
      bool is_appended = PyList_Append(current_list.borrowed(), tuple) == 0;
      Py_DECREF(tuple);
      if(not is_appended) return NULL;
    }
    if(i_point == i_point_end) // premature death. Retry with more atoms.
    {
      result.release(); // cleanup
      datapoints.clear(); // cleanup
      _natoms += _natoms >> 1; // increase number of atoms.
      return coordination_shells(_structure, _nshells, _center, _tolerance, _natoms);
    }
  }
  return result.release();
}
