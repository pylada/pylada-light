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

#ifdef PYLADA_INDEX
#  error PYLADA_INDEX already defined.
#endif
#define PYLADA_INDEX(a, b) (a(0) * b(1) + a(1)) * b(2) + a(2);
math::iVector3d guess_mesh_(Structure const &_structure, unsigned _nperbox)
{
  math::rMatrix3d const cell(math::gruber(_structure->cell));
  types::t_real const c0 = std::sqrt( cell.col(0).squaredNorm() );
  types::t_real const c1 = std::sqrt( cell.col(1).squaredNorm() );
  types::t_real const c2 = std::sqrt( cell.col(2).squaredNorm() );
  math::iVector3d result(1,1,1);

  size_t const N(_structure.size());
  size_t const Nboxes = size_t(std::floor(types::t_real(N)/types::t_real(_nperbox) + 0.5));
  if(Nboxes != 0) 
  {
    types::t_real mini = N / _nperbox * (c0+c1+c2);
    for(size_t n0(1); n0 <= Nboxes; ++n0)
      for(size_t n1(1); n1 <= Nboxes; ++n1)
      {
        size_t n2 = Nboxes / n0 / n1;
        if(n2 == 0) continue;
    
        types::t_real const a =
             std::abs(c0/types::t_real(n0) - c1/types::t_real(n1))
           + std::abs(c0/types::t_real(n0) - c2/types::t_real(n2))
           + std::abs(c1/types::t_real(n1) - c2/types::t_real(n2));
        if(a < mini) 
        {
          result(0) = n0;
          result(1) = n1;
          result(2) = n2;
          mini = a;
        }
      }
  }
  return result;
}
PyObject* dnc_boxes( const Structure &_structure, 
                            math::iVector3d const &_mesh, 
                            types::t_real _overlap)
{
  //namespace bt = boost::tuples;
  typedef math::iVector3d iVector3d;
  typedef math::rVector3d rVector3d;
  typedef math::rMatrix3d rMatrix3d;
  typedef Structure::const_iterator const_iterator;

  // constructs cell of small small box
  math::rMatrix3d const strcell( math::gruber(_structure.cell()) );
  math::rMatrix3d cell(strcell);
  for( size_t i(0); i < 3; ++i ) cell.col(i) /= types::t_real( _mesh(i) );
  
  // Inverse matrices.
  math::rMatrix3d const invstr(strcell.inverse()); // inverse of structure 
  math::rMatrix3d const invbox(cell.inverse());   // inverse of small box.

  // Edges
  rVector3d const sb_edges
    (
      _overlap / std::sqrt(cell.col(0).squaredNorm()),
      _overlap / std::sqrt(cell.col(1).squaredNorm()),
      _overlap / std::sqrt(cell.col(2).squaredNorm())
    );

  // Constructs mesh of small boxes.
  Py_ssize_t const Nboxes( _mesh(0) * _mesh(1) * _mesh(2) );
  python::Object container = PyList_New(Nboxes);
  for(Py_ssize_t i(0); i < Nboxes; ++i)
  {
    PyObject* newlist = PyList_New(0);
    if(not newlist) return NULL;
    PyList_SET_ITEM(container.borrowed(), i, newlist);
  }

  // Now adds points for each atom in each box.
  const_iterator i_atom = _structure.begin();
  const_iterator const i_atom_end = _structure.end();
  for( size_t index(0); i_atom != i_atom_end; ++i_atom, ++index )
  {
    // Gets coordinate in mesh of small-boxes. Only works because cell
    // and boxes are commensurate.
    rVector3d const rfrac(invbox*i_atom->pos());
    iVector3d const _ifrac(math::floor_int(rfrac));
    iVector3d const __ifrac( _ifrac(0) % _mesh(0),
                             _ifrac(1) % _mesh(1),
                             _ifrac(2) % _mesh(2) );
    iVector3d const ifrac
      (
        __ifrac(0) < 0 ? __ifrac(0) + _mesh(0): __ifrac(0), 
        __ifrac(1) < 0 ? __ifrac(1) + _mesh(1): __ifrac(1), 
        __ifrac(2) < 0 ? __ifrac(2) + _mesh(2): __ifrac(2)
      );
    // Computes index within cell of structure.
    types::t_int const u = PYLADA_INDEX(ifrac, _mesh);
#   ifdef PYLADA_DEBUG
      for(size_t i(0); i < 3; ++i)
        if(ifrac(i) < 0 or ifrac(i) >= _mesh(i))
        {
          PYLADA_PYERROR(InternalError, "Cell index out of range.");
          return NULL;
        }
      if(u < 0 or u >= Nboxes)
      {
        PYLADA_PYERROR(InternalError, "Cell index out of range.");
        return NULL;
      }
#   endif
    
    // creates apropriate point in small-box and adds it to list.
    math::rVector3d const orig_translation = cell * (ifrac - _ifrac).cast<math::rMatrix3d::Scalar>();
    { python::Object trans = python::numpy::wrap_to_numpy(orig_translation);
      if(not trans) return NULL;
      python::Object tuple = PyTuple_Pack(3, i_atom->borrowed(), trans.borrowed(), Py_True);
      if(not tuple) return NULL;
      PyObject* innerlist = PyList_GetItem(container.borrowed(), u);
      if(not innerlist) return NULL;
      if(PyList_Append(innerlist, tuple.borrowed()) != 0) return NULL; }
    
    // Finds out which other boxes it is contained in, including periodic images.
    for( types::t_int i(-1 ); i <= 1; ++i )
      for( types::t_int j(-1 ); j <= 1; ++j )
        for( types::t_int k(-1 ); k <= 1; ++k )
        {
          if( i == 0 and j == 0 and k == 0 ) continue;
    
          // First checks if on edge of small box.
#         ifndef PYLADA_WITH_EIGEN3
            rVector3d displaced = rfrac + sb_edges.cwise()*rVector3d(i,j,k);
#         else
            rVector3d displaced =   rfrac
                                  + (sb_edges.array()*rVector3d(i,j,k).array()).matrix();
#         endif
          iVector3d const boxfrac
            ( 
              math::floor_int(displaced(0)) == _ifrac(0) ? ifrac(0): ifrac(0) + i,
              math::floor_int(displaced(1)) == _ifrac(1) ? ifrac(1): ifrac(1) + j, 
              math::floor_int(displaced(2)) == _ifrac(2) ? ifrac(2): ifrac(2) + k
            );
          // Now checks if small box is at edge of periodic structure. 
          iVector3d const strfrac
            (
              boxfrac(0) < 0 ? 1: (boxfrac(0) >= _mesh(0) ? -1: 0),
              boxfrac(1) < 0 ? 1: (boxfrac(1) >= _mesh(1) ? -1: 0),
              boxfrac(2) < 0 ? 1: (boxfrac(2) >= _mesh(2) ? -1: 0)
            );
          bool const is_edge(strfrac(0) != 0 or strfrac(1) != 0 or strfrac(2) != 0);
      
          // Computes index of box where displaced atom is located.
          iVector3d const modboxfrac
            (
              boxfrac(0) + strfrac(0) * _mesh(0),
              boxfrac(1) + strfrac(1) * _mesh(1),
              boxfrac(2) + strfrac(2) * _mesh(2)
            );
          types::t_int const uu = PYLADA_INDEX(modboxfrac, _mesh);
#         ifdef PYLADA_DEBUG
            if(uu < 0 or uu >= types::t_int(PyList_Size(container.borrowed())))
            {
              PYLADA_PYERROR(InternalError, "Index out of range.");
              return NULL;
            }
#         endif
      
          // Don't need to go any further: not an edge state of either
          // small box or structure.
          if(u == uu  and not is_edge) continue;
      
          // Looks to see if object was already added.
          math::rVector3d const overlap_translation = 
               orig_translation + strcell*strfrac.cast<math::rMatrix3d::Scalar>();
          PyObject* innerlist = PyList_GetItem(container.borrowed(), uu);
          if(not innerlist) return NULL;
          Py_ssize_t const N = PyList_GET_SIZE(innerlist);
          bool found = false;
          for(Py_ssize_t i(0); i < N; ++i)
          {
            PyObject * const already_there = PyList_GET_ITEM(innerlist, i);
      
            if(    PyTuple_GET_ITEM(already_there, 2) != Py_False
                or PyTuple_GET_ITEM(already_there, 0) != i_atom->borrowed()) continue;
      
            PyObject* npyvec = PyTuple_GET_ITEM(already_there, 1);
            Eigen::Map< math::rVector3d > map( (math::rVector3d::Scalar*)
                                               PyArray_DATA((PyArrayObject*)npyvec) );
            if(math::eq(overlap_translation, map)) { found = true; break; }
          }
          if(not found)
          { // constructs overlap object and adds it to container.
            python::Object trans = python::numpy::wrap_to_numpy(overlap_translation);
            if(not trans) return NULL;
            python::Object overlap =PyTuple_Pack(3, i_atom->borrowed(), trans.borrowed(), Py_False); 
            if(not overlap) continue;
            if(PyList_Append(innerlist, overlap.borrowed()) != 0) return NULL; 
          }
        } // loops over next neighbor periodic images.
  }
  return container.release();
}
#undef PYLADA_INDEX
// \brief Wrapper to python for periodic boundary divide and conquer.
// \see  Pylada::crystal::periodic_dnc()
PyObject* pyperiodic_dnc(PyObject* _module, PyObject* _args, PyObject *_kwargs)
{
  PyObject* structure = NULL; 
  PyObject* _n = NULL;
  types::t_real overlap = 0e25;
  math::iVector3d mesh;
  long return_mesh = false;
  unsigned int nperbox = 0;
  static char *kwlist[] = { const_cast<char*>("structure"),
                            const_cast<char*>("overlap"), 
                            const_cast<char*>("mesh"), 
                            const_cast<char*>("nperbox"),
                            const_cast<char*>("return_mesh"), NULL};
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "Od|OIl:DnCBoxes", kwlist,
                                      &structure, &overlap, &_n, &nperbox, &return_mesh) )
    return NULL;
  if(_n != NULL and nperbox != 0)
  {
    PYLADA_PYERROR(TypeError, "DnCBoxes: Cannot specify both n and nperbox.");
    return NULL;
  }
  else if(_n == NULL) nperbox = 20;
  if(not check_structure(structure)) 
  {
    PYLADA_PYERROR(TypeError, "DnCBoxes: First argument should be a structure.");
    return NULL;
  }
  Structure struc = Structure::acquire(structure);
  if(_n != NULL and not python::numpy::convert_to_vector(_n, mesh)) return NULL;
  else if(_n == NULL) mesh = guess_mesh_(struc, nperbox);

  try
  { 
    python::Object result = dnc_boxes(struc, mesh, overlap);
    if(not result) return NULL;
    if(not return_mesh) return result.release();
    python::Object pymesh = python::numpy::wrap_to_numpy(mesh);
    if(not pymesh) return NULL;
    return PyTuple_Pack(2, pymesh.borrowed(), result.borrowed());
  }
  catch(...) {}
  return NULL;
}
