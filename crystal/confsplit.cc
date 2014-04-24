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

#ifdef PYLADA_GET_TRANS
#  error PYLADA_GET_TRANS already defined
#endif
#define PYLADA_GET_TRANS(NEIGH) \
    (math::rVector3d::Scalar*)PyArray_DATA( \
        (PyArrayObject*)PyTuple_GET_ITEM((PyTupleObject*)NEIGH, 1))
typedef Eigen::Map<math::rVector3d> t_NpMap;
typedef python::RAList_iterator pylist_iter;
typedef python::RATuple_iterator pytuple_iter;
math::rVector3d normalized(math::rVector3d const &_in) { return _in / std::sqrt(_in.squaredNorm()); }

//! Look for largest x element.
pylist_iter max_xelement( pylist_iter & _first,
                          pylist_iter const & _last,
                          math::rVector3d const &_x )
{
  if (_first == _last) return _first;
  do
  {
    t_NpMap const trans(PYLADA_GET_TRANS(*_first));
    if(not (math::is_null((_x - trans).squaredNorm()) or 
            math::is_null(_x.cross(trans).squaredNorm()))) break;
  }
  while (++_first != _last);
  if(_first == _last) return _last;
  pylist_iter _result = _first;
  t_NpMap resultpos(PYLADA_GET_TRANS(*_first));
  while (++_first != _last)
  {
    t_NpMap const vec(PYLADA_GET_TRANS(*_first));
    if(math::is_null( (_x - vec).squaredNorm())) continue;
    if(math::is_null(_x.cross(vec).squaredNorm())) continue;
    if(math::lt(resultpos.dot(_x), vec.dot(_x)))
    {
      _result = _first;
      new(&resultpos) t_NpMap(PYLADA_GET_TRANS(*_first));
    }
  }
  return _result;
}

//! Functor to compare coordinates using once given a basis.
struct CmpFromCoord
{
  math::rVector3d const &x;
  math::rVector3d const &y;
  math::rVector3d const &z;
  CmpFromCoord   (math::rVector3d const &_x, math::rVector3d const &_y, math::rVector3d const &_z) 
               : x(_x), y(_y), z(_z) {}
  CmpFromCoord(CmpFromCoord const &_in) : x(_in.x), y(_in.y), z(_in.z) {}
  bool operator()(PyObject* const _a, PyObject* const _b) const
  {
    t_NpMap const a(PYLADA_GET_TRANS(_a));
    t_NpMap const b(PYLADA_GET_TRANS(_b));
    const types::t_real x1(a.dot(x)), x2(b.dot(x));
    if( math::neq(x1, x2) ) return math::gt(x1, x2);
    const types::t_real y1(a.dot(y)), y2(b.dot(y));
    if( math::neq(y1, y2)) return math::gt(y1, y2);
    return math::gt(a.dot(z), b.dot(z) );
  }
};

//! Create tuple with atom and coordinates.
PyObject *create_tuple( PyObject *_neighbor, 
                        types::t_real const _x,
                        types::t_real const _y,
                        types::t_real const _z )
{
  python::Object result = PyTuple_New(2);
  if(not result) return NULL;
  // create numpy array.
  math::rVector3d const vec(_x, _y, _z);
  PyObject *array = python::numpy::wrap_to_numpy(vec);
  if(not array) return NULL;
  PyTuple_SET_ITEM(result.borrowed(), 1, (PyObject*)array);
  PyObject* const atom = PyTuple_GetItem(_neighbor, 0);
  Py_INCREF(atom);
  PyTuple_SET_ITEM(result.borrowed(), 0, atom);
  return result.release();
}
//! Converts bitset to configuration.
PyObject *convert_bitset( std::vector<PyObject*> const &_bitset,
                          math::rVector3d const &_x,
                          math::rVector3d const &_y,
                          math::rVector3d const &_z )
{
  python::Object result = PyTuple_New(_bitset.size());
  if(not result) return NULL;
  std::vector<PyObject*>::const_iterator i_first = _bitset.begin();
  std::vector<PyObject*>::const_iterator i_end = _bitset.end();
  for(size_t i(0); i_first != i_end; ++i_first, ++i)
  {
    t_NpMap const vec(PYLADA_GET_TRANS(*i_first));
    PyObject *item = create_tuple(*i_first, vec.dot(_x), vec.dot(_y), vec.dot(_z));
    if(not item) return NULL;
    PyTuple_SET_ITEM(result.borrowed(), i, item);
  }
  return result.release();
}

//! compare atomic types.
bool cmp_atom_types(PyObject *_a, PyObject *_b)
{
  return Atom::acquire(PyTuple_GET_ITEM(_a, 0)).type() 
           == Atom::acquire(PyTuple_GET_ITEM(_b, 0)).type();
}

// Compare new conf to old
bool cmp_to_confs( PyObject *const _config,
                   PyObject *const _bitset, 
                   types::t_real _tolerance )
{
  // compare configuration sizes.
  PyObject * const bitsetA = PyTuple_GET_ITEM(_config, 0);
  if(PyTuple_GET_SIZE(bitsetA) != PyTuple_GET_SIZE(_bitset)) return false;
  pytuple_iter i_b(_bitset, 0), i_a(bitsetA, 0);
  pytuple_iter i_end(_bitset);
  for(; i_b != i_end; ++i_b, ++i_a)
  {
    if(not cmp_atom_types(*i_a, *i_b)) return false; 
    t_NpMap const vecA(PYLADA_GET_TRANS(*i_a));
    t_NpMap const vecB(PYLADA_GET_TRANS(*i_b));
    if(math::neq(vecA(0), vecB(0), _tolerance)) return false;
    if(math::neq(vecA(1), vecB(1), _tolerance)) return false;
    if(math::neq(vecA(2), vecB(2), _tolerance)) return false;
  }
  return true;
}

bool splitconfigs( Structure const &_structure,
                   Atom const &_origin,
                   Py_ssize_t _nmax,
                   python::Object &_configurations,
                   types::t_real _tolerance )
{
  const types::t_real weight( 1e0 / types::t_real(_structure.size()) );

  // if configuration is null, creates it.
  if(not _configurations)
  {
    _configurations.reset(PyList_New(0));
    if(not _configurations) return false;
  }

  // creates bitset and first item.
  std::vector<PyObject*> bitset_list(_nmax);
  // holds ref until end of call.
  python::Object firstitem = PyTuple_New(3);
  if(not firstitem) return false;
  else
  {
    bitset_list[0] = firstitem.borrowed();
    PyTuple_SET_ITEM(firstitem.borrowed(), 0, _origin.new_ref());
    PyObject* pydist = PyFloat_FromDouble(0);
    if(not pydist) return false;
    PyTuple_SET_ITEM(firstitem.borrowed(), 2, pydist);
    math::rVector3d const zero = math::rVector3d::Zero();
    PyObject* pytrans = python::numpy::wrap_to_numpy(zero);
    if(not pytrans) return false;
    PyTuple_SET_ITEM(firstitem.borrowed(), 1, pytrans);
  }

  const math::rVector3d origin(_origin->pos);

  python::Object epositions = coordination_shells(_structure, _nmax, origin, _tolerance);

  // loop over epositions defining x.
  pylist_iter i_xpositions(epositions.borrowed(), 0);
  const types::t_real
    xweight( weight / types::t_real(PyList_GET_SIZE(*i_xpositions)) );
  pylist_iter i_xpos(*i_xpositions, 0);
  pylist_iter i_xpos_end(*i_xpositions);
  for(size_t n(0); i_xpos != i_xpos_end; ++i_xpos, ++n)
  {
    math::rVector3d const
      x(normalized(t_NpMap(PYLADA_GET_TRANS(*i_xpos))));

    // finds positions defining y.
    // Stores possible y positions.
    std::vector<PyObject*> ypossibles;
    pylist_iter i_ypositions = i_xpositions;
    if( PyList_GET_SIZE(*i_xpositions) == 1 ) ++i_ypositions; 

    pylist_iter max_x_element(i_ypositions);
    // might have to go to next shell for linear molecules.
    for(; i_ypositions != pylist_iter(epositions.borrowed()); ++i_ypositions) 
    {
      pylist_iter i_ypos(*i_ypositions, 0), i_ypos_end(*i_ypositions);
      max_x_element = max_xelement(i_ypos, i_ypos_end, x);
      if(max_x_element != i_ypos_end) break;
    }
    if(i_ypositions == pylist_iter(epositions.borrowed())) 
    {
      PYLADA_PYERROR(ValueError, "Pathological molecules. Could not determine y coordinate.");
      return false;
    }
    const types::t_real max_x_scalar_pos
      (t_NpMap(PYLADA_GET_TRANS(*max_x_element)).dot(x));
    pylist_iter i_ypos = pylist_iter(*i_ypositions, 0);
    pylist_iter const i_ypos_end = pylist_iter(*i_ypositions);
    for(; i_ypos != i_ypos_end; ++i_ypos)
    {
      t_NpMap const ypos(PYLADA_GET_TRANS(*i_ypos));
      if( math::neq(ypos.dot(x), max_x_scalar_pos) ) continue;
      if( math::is_null( (ypos - x).squaredNorm() ) ) continue;
      ypossibles.push_back(*i_ypos);
    }

    // divide current weight by number of possible y positions.
    types::t_real const bitsetweight = xweight / types::t_real(ypossibles.size());
    std::vector<PyObject*>::const_iterator i_ypossible = ypossibles.begin();
    std::vector<PyObject*>::const_iterator const i_ypossible_end = ypossibles.end();
    // loop over possible ys.
    //   Determine z from x and y.
    //   Basis is determined. Adds other atoms.
    for(; i_ypossible != i_ypossible_end; ++i_ypossible, ++n)
    {
      // at this point, we can define the complete coordinate system.
      const math::rVector3d yvec(PYLADA_GET_TRANS(*i_ypossible) );
      const math::rVector3d y(normalized(yvec - yvec.dot(x)*x));
      const math::rVector3d z( x.cross(y) );

      // atoms are now included in the list according to the following rule:
      //  _ closest to the origin first.
      //  _ ties are broken according to largest x coordinate.
      //  _ next ties are broken according to largest y coordinate.
      //  _ final ties are broken according to largest z coordinate.

      // we iterate over coordination shells and add reference until nmax is reached.
      Py_ssize_t current_index = 1;
      pylist_iter i_shell(epositions.borrowed(), 0);
      pylist_iter const i_shell_end(epositions.borrowed());
      for(; i_shell != i_shell_end; ++i_shell)
      {
        if( current_index == _nmax ) break;

        Py_ssize_t const N(PyList_GET_SIZE(*i_shell));
        if(N == 1) // case where the shell contains only one atom ref.
        {
          bitset_list[current_index++] = PyList_GET_ITEM(*i_shell, 0);
          continue;
        }
        Py_ssize_t const edn(std::min(N, _nmax - current_index));

        // copy list to vector for sorting.
        std::vector<PyObject*> sortme(N);
        std::copy(pylist_iter(*i_shell, 0), pylist_iter(*i_shell), sortme.begin());
        // case where all atom in shell should be added.
        if(edn == N) 
          std::sort(sortme.begin(), sortme.end(), CmpFromCoord(x, y, z));
        // case where only a few atoms in the shell should be added.
        else
          std::partial_sort( sortme.begin(), sortme.begin() + edn, sortme.end(), 
                             CmpFromCoord(x, y, z) );
        std::copy(sortme.begin(), sortme.begin() + edn, bitset_list.begin()+current_index);
        current_index += edn;
      } // end of loop over positions at equivalent distance.


      // finally adds configuration.
      python::Object pybitset = convert_bitset(bitset_list, x, y, z);
      pylist_iter const i_conf_end(_configurations.borrowed());
      pylist_iter i_found = std::find_if
        (
          pylist_iter(_configurations.borrowed(), 0), i_conf_end,
          boost::bind(cmp_to_confs, _1, pybitset.borrowed(), _tolerance)
        );
      if(i_found == i_conf_end) // add new bitset.
      {
        python::Object pyconf = PyTuple_New(2);
        if(not pyconf) return false;
        PyObject *pyweight = PyFloat_FromDouble(bitsetweight);
        if(not pyweight) return false;
        PyTuple_SET_ITEM(pyconf.borrowed(), 0, pybitset.new_ref());
        PyTuple_SET_ITEM(pyconf.borrowed(), 1, pyweight);
        // should not have been set yet.
        if( PyList_Append(_configurations.borrowed(), pyconf.borrowed()) != 0)
          return false;
      }
      else // add to pre-existing bitset.
      {
        PyObject * pyreal = PyTuple_GET_ITEM(*i_found, 1);
        double const real = PyFloat_AS_DOUBLE(pyreal);
        if(PyErr_Occurred() != NULL) return false;
        PyObject *dummy = PyFloat_FromDouble(bitsetweight+real); 
        if(not dummy) return false;
        PyTuple_SET_ITEM(*i_found, 1, dummy);
        Py_DECREF(pyreal);
      }
    } // end of loop over equivalent y coords.
  } // end of loop over equivalent  x coords.
  return true;
}
#undef PYLADA_GET_TRANS
