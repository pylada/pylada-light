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

//! Returns the primitive unit structure. 
Structure primitive(Structure const &_structure, types::t_real _tolerance)
{
  if( _tolerance < 0e0 ) _tolerance = types::tolerance;
  if( _structure.size() == 0 )
  {
    PYLADA_PYERROR(ValueError, "Empty structure.");
    return Structure();
  }

  // copies lattice.
  Structure result(_structure.copy());
  if(not result)
  {
    PYLADA_PYERROR(ValueError, "primitive: Input lattice is not deep-copyable.");
    return Structure();
  }
  math::rMatrix3d const cell = math::gruber(result.cell());
  bool is_primitive = true;

  // moves sites into unit-cell.
  math::rMatrix3d const inv(cell.inverse());
  Structure::iterator i_atom = result.begin();
  Structure::iterator const i_atom_end = result.end();
  for(; i_atom != i_atom_end; ++i_atom)
    i_atom->pos() = into_cell(i_atom->pos(), cell, inv);

  // Then compares fractional translations from site 0 to sites of same type.
  std::vector<math::rVector3d> translations;
  python::Object const front_type(result.front().type());
  math::rVector3d const center = result.front()->pos;
  Structure::const_iterator i_site = _structure.begin();
  Structure::const_iterator const i_site_end = _structure.end();
  for(; i_site != i_site_end; ++i_site )
  {
    // Translations are created from equivalent sites only.
    if(front_type != i_site->type()) continue;

    // creates translation.
    math::rVector3d const translation = into_voronoi(i_site->pos() - center, cell, inv);
    
    // loop on null translation.
    if( math::is_null(translation, _tolerance) ) continue;

    // checks that it leaves the lattice invariant.
    Structure::const_iterator i_mapping = _structure.begin();
    Structure::const_iterator const i_fend = result.end(); 
    for(; i_mapping != i_site_end; ++i_mapping)
    {
      math::rVector3d const pos = into_cell(i_mapping->pos() + translation, cell, inv);
      Structure::iterator i_found = result.begin(); 
      for(; i_found != i_fend; ++i_found)
        if( math::eq(pos, i_found->pos(), _tolerance) and i_mapping->type() == i_found->type() ) break;
      if(i_found == i_fend) break;
    }

    if( i_mapping != i_site_end ) continue;

    // adds translation to vector. This lattice is not primitive.
    translations.push_back(into_voronoi(translation, cell));
    is_primitive = false;
  }

  // This lattice is primitive.
  if( is_primitive ) return result;

  // adds original translations.
  translations.push_back( cell.col(0) );
  translations.push_back( cell.col(1) );
  translations.push_back( cell.col(2) );

  // Loops over possible primitive cells.
  typedef std::vector<math::rVector3d> :: const_iterator t_cit;
  t_cit const i_vec_begin( translations.begin() );
  t_cit const i_vec_end( translations.end() );
  math::rMatrix3d new_cell = result.cell();
  types::t_real volume = std::abs(new_cell.determinant());
  for( t_cit i_first(i_vec_begin); i_first != i_vec_end; ++i_first )
    for( t_cit i_second(i_vec_begin); i_second != i_vec_end; ++i_second )
    {
      if( i_first == i_second ) continue;
      for( t_cit i_third(i_vec_begin); i_third != i_vec_end; ++i_third )
      {
        if( i_first == i_third or i_second == i_third ) continue;
        // construct new cell.
        math::rMatrix3d trial;
        trial.col(0) = *i_first;
        trial.col(1) = *i_second;
        trial.col(2) = *i_third;

        // singular matrix?
        types::t_real const det(trial.determinant());
        if( math::is_null(det, 3e0*_tolerance) ) continue;
        // Volume smaller than current new_cell?
        if( math::geq(std::abs(det), volume, 3e0 * _tolerance) ) continue;
        // Direct matrix?
        if( det < 0e0 )
        {
          trial.col(2) = *i_second;
          trial.col(1) = *i_third;
#         ifdef PYLADA_DEBUG
            if(trial.determinant() < types::tolerance)
            {
              PYLADA_PYERROR(internal, "Negative volume.");
              return Structure();
            }
#         endif
        }
        // Checks that original cell is a supercell.
        if( not math::is_integer(trial.inverse() * result.cell(), _tolerance) ) continue;

        // Checks that all lattice sites are integers.
        volume = std::abs(det);
        new_cell = trial;
      }
    }

  // Found the new cell with smallest volume (e.g. primivite)
  if(math::eq(_structure.volume(), new_cell.determinant()))
  {
    PYLADA_PYERROR(internal, "Found translation but no primitive cell.");
    return Structure();
  }

  // now creates new lattice.
  result.clear();
  result.cell() = math::gruber(new_cell);
  math::rMatrix3d const inv_cell(result.cell().inverse());
  for(i_site = _structure.begin(); i_site != i_site_end; ++i_site)
  {
    math::rVector3d const pos = into_cell(i_site->pos(), result.cell(), inv_cell); 
    Structure::const_iterator i_found = result.begin(); 
    Structure::const_iterator const i_fend = result.end(); 
    for(; i_found != i_fend; ++i_found)
      if(math::eq(i_found->pos(), pos) and i_site->type() == i_found->type()) break;
    if( i_found == i_fend )
    {
      result.push_back(i_site->copy());
      result.back()->pos = pos;
    }
  }
  if(_structure.size() % result.size() != 0)
  {
    PYLADA_PYERROR(internal, "Nb of atoms in output not multiple of input.");
    return Structure();
  }
  if(math::neq(types::t_real(_structure.size()/result.size()), _structure.volume()/result.volume()))
  {
    PYLADA_PYERROR(internal, "Size and volumes do not match.");
    return Structure();
  }

  return result;
}


//! Returns the primitive unit structure. 
bool is_primitive(Structure const &_structure, types::t_real _tolerance)
{
  if( _tolerance < 0e0 ) _tolerance = types::tolerance;
  if( _structure.size() == 0 )
  { 
    PYLADA_PYERROR(ValueError, "Empty structure.");
    //BOOST_THROW_EXCEPTION(error::internal() << error::string("empty structure"));
    BOOST_THROW_EXCEPTION(error::internal());
  }

  // copies lattice.
  math::rMatrix3d const cell = math::gruber(_structure.cell());

  // moves sites into unit-cell.
  math::rMatrix3d const inv(cell.inverse());

  // Then compares fractional translations from site 0 to sites of same type.
  python::Object const front_type(_structure.front().type());
  math::rVector3d const center = into_cell(_structure.front().pos(), cell, inv);
  Structure::const_iterator i_site = _structure.begin();
  Structure::const_iterator const i_site_end = _structure.end();
  for(; i_site != i_site_end; ++i_site)
  {
    // Translations are created from equivalent sites only.
    if(front_type != i_site->type()) continue;

    // creates translation.
    math::rVector3d const translation = into_voronoi(i_site->pos() - center, cell, inv);
    
    // loop on null translation.
    if( math::is_null(translation, _tolerance) ) continue;

    // checks that it leaves the lattice invariant.
    Structure::const_iterator i_mapping = _structure.begin();
    Structure::const_iterator const i_fend = _structure.end(); 
    for(; i_mapping != i_site_end; ++i_mapping)
    {
      math::rVector3d const pos = into_cell(i_mapping->pos() + translation, cell, inv);
      Structure::const_iterator i_found = _structure.begin(); 
      for(; i_found != i_fend; ++i_found)
        if( math::are_periodic_images(pos, i_found->pos(), inv, _tolerance) 
            and i_mapping->type() == i_found->type() ) break;
      if(i_found == i_fend) break;
    }

    if( i_mapping == i_site_end ) return false;
  }
  return true;
}
