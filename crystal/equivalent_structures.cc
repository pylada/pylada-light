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

//! \brief Returns true if two structures are equivalent lattices in same cartesian coordinates. 
//! \details Two structures are equivalent in a mathematical sense, for the
//!          same cartesian coordinates. That is, all sites defined by one
//!          lattice correspond to the same site in the other lattice.
//!          Supercells are *not* equivalent to their parent lattice (use
//!          in conjunction with primitive() to get this behavior). Note that since the carete
//! \param[in] _a: The first structure.
//! \param[in] _b: The second structure.
//! \param[in] with_scale: whether to take the scale into account. Defaults to true.
//! \param[in] tolerance: Tolerance when comparing distances. Defaults to
//!            types::t_real. It is in the same units as the structures scales, if
//!            that is taken into account, otherwise, it is in the same
//!            units as _a.scale.
bool equivalent_lattices( Structure const &_a, Structure const &_b,
                          bool with_scale, types::t_real _tol = types::tolerance )
{
  // different number of atoms.
  if(_a.size() != _b.size()) return false;
  types::t_real const scaleA = _a.scale();
  types::t_real const scaleB
    = with_scale ?
        _b.scale():
        _a.scale() * std::pow(std::abs(_a.cell().determinant() / _b.cell().determinant()), 1e0/3e0);
  // different volume.
  if(math::neq( std::abs((_a.cell()*scaleA).determinant()),
                std::abs((_b.cell()*scaleB).determinant()), 3e0*_tol)) return false;
  // different lattice parameterization.
  if(not math::is_integer(_a.cell().inverse() * _b.cell(), 3e0*_tol)) return false;
  if(not math::is_integer(_b.cell().inverse() * _a.cell(), 3e0*_tol)) return false;
  
  // check possible rotation. 
  math::rMatrix3d const cellA = math::gruber(_a.cell(), 100, _tol) * scaleA;
  math::rMatrix3d const invA = cellA.inverse();
  
  // creates a vector referencing A atomic sites.
  // Items from this list will be removed as they are found.
  typedef std::list<size_t> t_List;
  t_List atomsA;
  for(size_t i(0); i < _a.size(); ++i) atomsA.push_back(i);

  Structure::const_iterator i_b = _b.begin();
  Structure::const_iterator const i_bend = _b.end();
  for(; i_b != i_bend; ++i_b)
  {
    math::rVector3d const pos = into_voronoi(i_b->pos()*scaleB, cellA, invA);
    t_List :: iterator i_first =  atomsA.begin();
    t_List :: iterator const i_end = atomsA.end();
    if(i_first == i_end) return false;
    for(; i_first != i_end; ++i_first)
    {
      if( not math::is_integer(invA * (pos - _a[*i_first]->pos*scaleA), 4*_tol) ) continue;
      if( i_b->type() == _a[*i_first]->type ) break;
    }
    if(i_first == i_end) break;
    atomsA.erase(i_first);
  }
  return i_b == i_bend;
}

//! Finds atom accounting for least number of similar type.
int min_test(Structure const &_a)
{
  std::vector<size_t> mini(_a.size(), 1);
  std::vector<size_t>::iterator i_fmin = mini.begin();
  Structure::const_iterator i_first = _a.begin();
  Structure::const_iterator const i_end = _a.end();
  for(; i_first != i_end; ++i_first, ++i_fmin)
  {
    if(*i_fmin > 1) continue;
    Structure::const_iterator i_second = i_first + 1;
    std::vector<size_t>::iterator i_smin = i_fmin + 1;
    for(; i_second != i_end; ++i_second, ++i_smin)
      if(i_first->type() == i_second->type()) ++(*i_fmin), ++(*i_smin);
  }
  return std::min_element(mini.begin(), mini.end()) - mini.begin();
}

//! \brief Returns true if two structures are equivalent. 
//! \details Two structures are equivalent in a crystallographic sense,
//!          e.g. without reference to cartesian coordinates or possible
//!          motif rotations which leave the lattice itself invariant. A
//!          supercell is *not* equivalent to its lattice, unless it is a
//!          trivial supercell.
//! \param[in] _a: The first structure.
//! \param[in] _b: The second structure.
//! \param[in] with_scale: whether to take the scale into account. Defaults to true.
//! \param[in] tolerance: Tolerance when comparing distances. Defaults to
//!            types::t_real. It is in the same units as the structures scales, if
//!            that is taken into account, otherwise, it is in the same
//!            units as _a.scale.
bool equivalent_crystals( Structure const &_a, Structure const &_b,
                          bool with_scale, types::t_real _tol )
{
  // different number of atoms.
  if(_a.size() != _b.size()) return false;
  types::t_real const scaleA = _a.scale();
  types::t_real const scaleB
    = with_scale ?
        _b.scale():
        _a.scale() * std::pow(std::abs(_a.cell().determinant() / _b.cell().determinant()), 1e0/3e0);
  // different volume.
  if(math::neq( std::abs((_a.cell()*scaleA).determinant()),
                std::abs((_b.cell()*scaleB).determinant()), 3e0*_tol)) return false;
  
  // check possible rotation. 
  math::rMatrix3d const cellA = math::gruber(_a.cell(), 100, _tol) * scaleA;
  math::rMatrix3d const cellB = math::gruber(_b.cell(), 100, _tol) * scaleB;
  math::rMatrix3d const invA = cellA.inverse();
  math::rMatrix3d const invB = cellB.inverse();
  math::rMatrix3d const rot = cellA * cellB.inverse();
  if(not math::is_identity(rot * (~rot), 2*_tol)) return false;
  if(math::neq(rot.determinant(), 1e0, 3*_tol))  return false;
  
  // Now checks atomic sites. 
  // first computes point-group symmetries.
  python::Object pg = cell_invariants(cellA);
  // then find the occupation type with the smallest number of occurences.
  python::Object mintype = _a[min_test(_a)].type();
  
  // Computes possible translations, looking at only one type of site-occupation.
  // The center of gravity will tell us about a possible translation of
  // the cartesian basis.
  math::rVector3d transA(0,0,0);
  size_t nA(0);
  Structure::const_iterator i_atom = _a.begin(); 
  Structure::const_iterator i_atom_end = _a.end(); 
  for(; i_atom != i_atom_end; ++i_atom)
    if(mintype == i_atom->type())
    {
      transA += into_voronoi(i_atom->pos() * scaleA, cellA, invA);
      ++nA;
    }
  transA /= types::t_real(nA);
  math::rVector3d transB(0,0,0);
  size_t nB = 0;
  for(i_atom = _b.begin(), i_atom_end =_b.end(); i_atom != i_atom_end; ++i_atom)
    if(mintype == i_atom->type())
    {
      transB += into_voronoi(i_atom->pos() * scaleB, cellB, invB);
      ++nB;
      if(nB > nA) return false;
    }
  transB /= types::t_real(nB);

  // loop over possible motif rotations.
  python::Object iter_op = PyObject_GetIter(pg.borrowed());
  for( python::Object symop(PyIter_Next(iter_op.borrowed()));
       symop.is_valid();
       symop.reset(PyIter_Next(iter_op.borrowed())) ) 
  {
    types::t_real * const symop_data = (types::t_real*)PyArray_DATA((PyArrayObject*)symop.borrowed());
    Eigen::Map< Eigen::Matrix<types::t_real, 4, 3> > opmap(symop_data);
    // creates a vector referencing B atomic sites.
    // Items from this list will be removed as they are found.
    std::list<size_t> atomsA;
    for(size_t i(0); i < _a.size(); ++i) atomsA.push_back(i);

    math::rMatrix3d const rotation = opmap.block<3,3>(0,0) * rot;
    
    Structure::const_iterator i_b = _b.begin();
    Structure::const_iterator const i_bend = _b.end();
    for(; i_b != i_bend; ++i_b)
    {
      math::rVector3d const pos = rotation * (into_voronoi(i_b->pos()*scaleB, cellB, invB) - transB);
      std::list<size_t>::iterator i_first =  atomsA.begin();
      std::list<size_t>::iterator const i_end = atomsA.end();
      if(i_first == i_end) return false;
      for(; i_first != i_end; ++i_first)
      {
        if( not math::is_integer(invA * (pos - _a[*i_first]->pos*scaleA + transA), 4*_tol) ) continue;
        if( i_b->type() == _a[*i_first]->type ) break;
      }
      if(i_first == i_end)  break; 
      atomsA.erase(i_first);
    }
    if(i_b == i_bend) return true;
  }
  return false;
}

//! \brief Returns true if two structures are equivalent. 
//! \details Two structures are equivalent in a crystallographic sense,
//!          e.g. without reference to cartesian coordinates or possible
//!          motif rotations which leave the lattice itself invariant. A
//!          supercell is *not* equivalent to its lattice, unless it is a
//!          trivial supercell.
//! \param[in] _a: The first structure.
//! \param[in] _b: The second structure.
//! \param[in] with_scale: whether to take the scale into account. Defaults to true.
//! \param[in] tolerance: Tolerance when comparing distances. Defaults to
//!            types::t_real. It is in the same units as the structures scales, if
//!            that is taken into account, otherwise, it is in the same
//!            units as _a.scale.
bool equivalent( Structure const &_a, Structure const &_b,
                 bool with_scale, bool with_cartesian, types::t_real _tol )
{
  if(with_cartesian)
  {
    PYLADA_PYERROR(NotImplementedError, "Equivalence in different cartesian coordinates not implemented.");
    return false;
  }
  return with_cartesian ? equivalent_crystals(_a, _b, with_scale, _tol):
                          equivalent_lattices(_a, _b, with_scale, _tol);
}
