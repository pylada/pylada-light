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

template<size_t N0, size_t N1>  PyObject* new_matrix()
{
  npy_intp dims[2] = {N0, N1};
  int const type = python::numpy::type<types::t_real>::value;
  PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(2, dims, type, 1);
  return (PyObject*)result;
}
template<size_t N0, size_t N1>  PyObject* new_identity()
{
  PyArrayObject *result = (PyArrayObject*) new_matrix<N0, N1>();
  if(not result) return NULL;
  for(size_t i(0); i < std::min(N0, N1); ++i)
    *((types::t_real*)PyArray_GETPTR2(result, i, i)) = 1;
  return (PyObject*)result;
}

// \brief Finds and stores point group operations.
// \details Rotations are determined from G-vector triplets with the same
//          norm as the unit-cell vectors.
// \param[in] _structure The structure for which to find the point group.
// \param[in] _tol acceptable tolerance when determining symmetries.
//             -1 implies that types::tolerance is used.
// \retval python list of affine symmetry operations for the given structure.
// \see Taken from Enum code, PRB 77, 224115 (2008).
PyObject* cell_invariants(math::rMatrix3d const &_cell, types::t_real _tolerance)
{
  if( _tolerance <= 0e0 ) _tolerance = types::tolerance;
  python::Object result = PyList_New(0);
  if(not result) return NULL;
  PyObject * const identity = new_identity<4, 3>();
  if(not identity) return NULL;

  // Now append object to list.
  if(PyList_Append(result.borrowed(), identity) != 0) { Py_DECREF(identity); return NULL; }
  Py_DECREF(identity);
  
  // Finds out how far to look.
  types::t_real const volume( std::abs(_cell.determinant()) );
  math::rVector3d const a0( _cell.col(0) );
  math::rVector3d const a1( _cell.col(1) );
  math::rVector3d const a2( _cell.col(2) );
  types::t_real const max_norm = std::max( a0.norm(), std::max(a1.norm(), a2.norm()) );
  int const n0( std::ceil(max_norm*(a1^a2).norm()/volume) );
  int const n1( std::ceil(max_norm*(a2^a0).norm()/volume) );
  int const n2( std::ceil(max_norm*(a0^a1).norm()/volume) );
  types::t_real const length_a0( a0.squaredNorm() );
  types::t_real const length_a1( a1.squaredNorm() );
  types::t_real const length_a2( a2.squaredNorm() );

  // now creates a vector of all G-vectors in the sphere of radius max_norm. 
  typedef std::vector<math::rVector3d, Eigen::aligned_allocator<math::rVector3d> > t_vector;
  t_vector gvectors[3];
  for( int i0(-n0); i0 <= n0; ++i0 )
    for( int i1(-n1); i1 <= n1; ++i1 )
      for( int i2(-n2); i2 <= n2; ++i2 )
      {
        math::rVector3d const g = _cell * math::rVector3d(i0, i1, i2);
        types::t_real length( g.squaredNorm() );
        if( std::abs(length-length_a0) < _tolerance ) gvectors[0].push_back(g); 
        if( std::abs(length-length_a1) < _tolerance ) gvectors[1].push_back(g); 
        if( std::abs(length-length_a2) < _tolerance ) gvectors[2].push_back(g); 
      }


  // Adds triplets which are rotations.
  math::rMatrix3d const inv_cell(_cell.inverse());
  t_vector::const_iterator i_a0 = gvectors[0].begin();
  t_vector::const_iterator const i_a0_end = gvectors[0].end();
  t_vector::const_iterator const i_a1_end = gvectors[1].end();
  t_vector::const_iterator const i_a2_end = gvectors[2].end();
  for(; i_a0 != i_a0_end; ++i_a0)
  {
    t_vector::const_iterator i_a1 = gvectors[1].begin();
    for(; i_a1 != i_a1_end; ++i_a1)
    {
      t_vector::const_iterator i_a2 = gvectors[2].begin();
      for(; i_a2 != i_a2_end; ++i_a2)
      {
        // creates matrix.
        math::rMatrix3d rotation;
        rotation.col(0) = *i_a0;
        rotation.col(1) = *i_a1;
        rotation.col(2) = *i_a2;

        // checks that this the rotation is not singular.
        if( math::is_null(rotation.determinant(), _tolerance) ) continue;

        rotation = rotation * inv_cell;
        // avoids identity.
        if( math::is_identity(rotation, _tolerance) ) continue;
        // checks that the rotation is a rotation.
        if( not math::is_identity(rotation * (~rotation), _tolerance) ) continue;

        // Check if operation is new.
        bool doadd = true;
        python::Object iterator = PyObject_GetIter(result.borrowed());
        if(not iterator) { return NULL; }
        python::Object item(PyIter_Next(iterator.borrowed()));
        for(; item.is_valid() and doadd; item.reset(PyIter_Next(iterator.borrowed())))
        {
          // reinitializes operation map.
          types::t_real *const symop
            = (types::t_real*)PyArray_DATA((PyArrayObject*)item.borrowed());
          Eigen::Map<math::rMatrix3d> rotmap(symop);
          doadd = math::neq(rotmap, rotation, _tolerance);
        }
        if(not doadd) continue;

        // adds to vector of symmetries.
        python::Object symop = new_matrix<4, 3>();
        if(not symop) return NULL;
        types::t_real * const symop_ = 
          (types::t_real*)PyArray_DATA((PyArrayObject*)symop.borrowed());
        Eigen::Map< Eigen::Matrix<types::t_real, 4, 3> > opmap(symop_);
        opmap.block<3,3>(0,0) = rotation;
        opmap.block<1,3>(3,0) = math::rVector3d::Zero();
      
        if(PyList_Append(result.borrowed(), symop.borrowed()) != 0) return NULL;
      } // a2
    } // a1
  } // a0
  return result.release();
}

// \brief Finds and stores space group operations.
// \param[in] _structure The structure for which to find the space group.
// \param[in] _tol acceptable tolerance when determining symmetries.
//             -1 implies that types::tolerance is used.
// \retval spacegroup python list of symmetry operations for the given structure.
// \warning Works for primitive lattices only.
// \see Taken from Enum code, PRB 77, 224115 (2008).
PyObject* space_group(Structure const &_lattice, types::t_real _tolerance)
{
  if(_tolerance <= 0e0) _tolerance = types::tolerance;
  // Checks that lattice has atoms.
  if(_lattice.size() == 0) 
  {
    PYLADA_PYERROR(ValueError, "space_group: Input lattice is empty.");
    return NULL;
  }
  // Checks that lattice is primitive.
  if(not is_primitive(_lattice, _tolerance)) 
  {
    PYLADA_PYERROR(ValueError, "space_group: Input lattice is not primitive.");
    return NULL;
  }
  
  // Finds minimum translation.
  math::rVector3d translation(_lattice.front()->pos);
  math::rMatrix3d const cell(math::gruber(_lattice.cell()));
  math::rMatrix3d const invcell(!cell);
  // Creates a list of atoms centered in the cell.
  Structure::t_Atoms atoms; atoms.reserve(_lattice.size());
  Structure::const_iterator i_site = _lattice.begin();
  Structure::const_iterator const i_site_end = _lattice.end();
  for(; i_site != i_site_end; ++i_site)
  {
    Atom atom;
    atom.pos() = into_cell(i_site->pos()-translation, cell, invcell);
    atom.type(i_site->type());
    atoms.push_back(atom);
  }

  // gets point group.
  python::Object pg = cell_invariants(_lattice.cell());
  if(not pg) return NULL;
  if(PyList_Size(pg.borrowed()) == 0) 
  {
    PYLADA_PYERROR(InternalError, "Point-group is unexpectedly empty.");
    return NULL;
  }
  python::Object result = PyList_New(0);
  if(not result) return NULL;
       
  // lists atoms of same type as atoms.front()
  std::vector<math::rVector3d> translations;
  python::Object const fronttype(atoms.front().type());
  Structure::t_Atoms::const_iterator const i_atom_begin = atoms.begin();
  Structure::t_Atoms::const_iterator const i_atom_end = atoms.end();
  for(Structure::t_Atoms::const_iterator i_atom = i_atom_begin; i_atom != i_atom_end; ++i_atom)
    if(fronttype == i_atom->type()) translations.push_back(i_atom->pos());
  

  // applies point group symmetries and finds out if they are part of the space-group.
  python::Object const i_op = PyObject_GetIter(pg.borrowed());
  if(not i_op) return NULL;
  python::Object symop = PyIter_Next(i_op.borrowed());
  for( ; symop.is_valid(); symop.reset(PyIter_Next(i_op.borrowed())))
  {
    types::t_real * const symop_ = (types::t_real*)PyArray_DATA((PyArrayObject*)symop.borrowed());
    Eigen::Map< Eigen::Matrix<types::t_real, 4, 3> > opmap(symop_);
    // loop over possible translations.
    std::vector<math::rVector3d> :: const_iterator i_trial = translations.begin();
    std::vector<math::rVector3d> :: const_iterator const i_trial_end = translations.end();
    for(; i_trial != i_trial_end; ++i_trial)
    {
      // Checks that this is a mapping of the lattice upon itself.
      Structure::t_Atoms::const_iterator i_unmapped = i_atom_begin;
      for(; i_unmapped != i_atom_end; ++i_unmapped)
      {
        math::rVector3d const transpos
          = into_cell( opmap.block<3,3>(0, 0)*i_unmapped->pos() + (*i_trial), cell, invcell );
        Structure::t_Atoms::const_iterator i_mapping = i_atom_begin;
        for(; i_mapping != i_atom_end; ++i_mapping)
          if(     math::eq(transpos, i_mapping->pos(), _tolerance) 
              and i_unmapped->type() == i_mapping->type() ) break;
        // found unmapped site if condition is true.
        if(i_mapping == i_atom_end) break;
      } // loop over all atoms.

      // all sites in the lattice were mapped if condition is true.
      if(i_unmapped == i_atom_end) break; 
    } // loop over trial translations.

    // Found transformation which maps lattice upon itself if condition is true.
    if(i_trial != i_trial_end)
    {
      // set translation of symmetry operator.
      opmap.block<1, 3>(3, 0)
        = into_voronoi( *i_trial - opmap.block<3,3>(0,0) * translation + translation,
                        cell, invcell );
      // append to list.
      if(PyList_Append(result.borrowed(), symop.borrowed()) != 0) return NULL;
    }
  } // loop over point group.

  return result.release();
} 
