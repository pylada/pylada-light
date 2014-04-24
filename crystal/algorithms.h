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
  //! \brief Map atomic sites from mapper onto mappee.
  //! \param[in] _mapper : a lattice against which to map atomic sites.
  //! \param[inout] _mappee : a supercell for which sites will be mapped.
  //! \param[in] _withocc : whether to take occupation into count, or only position.
  //! \param[in] _tolerance : Tolerance criteria for distances, in units of _mappee.scale().
  //! \details Where possible, the site indices of the mappee structure
  //!          corresponds to the equivalent sites in the mapper structure.
  //! \return True if mapping is successful, False if all sites could not be mapped. 
  //!         Since in the case of defects, incomplete mappings may be what is wanted, 
  PYLADA_INLINE bool map_sites( Structure const &_mapper,
                              Structure &_mappee,
                              python::Object _withocc = python::Object(),
                              types::t_real _tolerance = types::tolerance )
    PYLADA_END(return (*(bool(*)( Structure const&,
                                Structure &, 
                                python::Object, 
                                types::t_real ))
                     api_capsule[PYLADA_SLOT(crystal)])
                    (_mapper, _mappee, _withocc, _tolerance);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)map_sites;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Returns true if two structures are equivalent. 
  //! \details Two structures are equivalent in a crystallographic sense,
  //!          e.g. without reference to cartesian coordinates or possible
  //!          motif rotations which leave the lattice itself invariant. A
  //!          supercell is *not* equivalent to its lattice, unless it is a
  //!          trivial supercell.
  //! \param[in] _a: The first structure.
  //! \param[in] _b: The second structure.
  //! \param[in] scale: whether to take the scale into account. Defaults to true.
  //! \param[in] cartesian: whether to take into account differences in
  //!            cartesian coordinates. Defaults to true. If False, then
  //!            comparison is according to mathematical definition of a
  //!            lattice. If True, comparison is according to
  //!            crystallographic comparison.
  //! \param[in] tolerance: Tolerance when comparing distances. Defaults to
  //!            types::t_real. It is in the same units as the structures scales, if
  //!            that is taken into account, otherwise, it is in the same
  //!            units as _a.scale.
  PYLADA_INLINE bool equivalent( Structure const &_a, 
                               Structure const &_b,
                               bool with_scale=true, 
                               bool with_cartesian=true,
                               types::t_real _tol = types::tolerance )
    PYLADA_END(return (*(bool(*)( Structure const&, 
                                Structure const &, 
                                bool, 
                                bool, 
                                types::t_real ))
                     api_capsule[PYLADA_SLOT(crystal)])
                    (_a, _b, with_scale, with_cartesian, _tol);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)equivalent;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Returns the primitive unit structure. 
  PYLADA_INLINE Structure primitive( Structure const &_structure, 
                                   types::t_real _tolerance = -1e0 )
      PYLADA_END(return (*(Structure(*)( Structure const&, 
                                       types::t_real ))
                       api_capsule[PYLADA_SLOT(crystal)])
                      (_structure, _tolerance);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *) primitive;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Returns True if the input is primitive.
  PYLADA_INLINE bool is_primitive( Structure const &_structure, 
                                 types::t_real _tolerance = -1e0 )
    PYLADA_END(return (*(bool(*)(Structure const&, types::t_real))
                     api_capsule[PYLADA_SLOT(crystal)])(_structure, _tolerance);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)is_primitive;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)


#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Finds and stores point group operations.
  //! \details Rotations are determined from G-vector triplets with the same
  //!          norm as the unit-cell vectors.
  //! \param[in] cell The cell for which to find the point group.
  //! \param[in] _tol acceptable tolerance when determining symmetries.
  //!             -1 implies that types::tolerance is used.
  //! \retval python list of affine symmetry operations for the given structure.
  //!         Each element is a 4x3 numpy array, with the first 3 rows
  //!         forming the rotation, and the last row is the translation.
  //!         The affine transform is applied as rotation * vector + translation.
  //!         `cell_invariants` always returns isometries (translation is zero).
  //! \see Taken from Enum code, PRB 77, 224115 (2008).
  PYLADA_INLINE PyObject* cell_invariants( math::rMatrix3d const &_cell, 
                                         types::t_real _tolerance = -1e0 )
    PYLADA_END(return (*(PyObject*(*)(math::rMatrix3d const &, types::t_real))
                     api_capsule[PYLADA_SLOT(crystal)])(_cell, _tolerance);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)cell_invariants;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Finds and stores space group operations.
  //! \param[in] _structure The structure for which to find the space group.
  //! \param[in] _tol acceptable tolerance when determining symmetries.
  //!             -1 implies that types::tolerance is used.
  //! \retval spacegroup python list of symmetry operations for the given structure.
  //!         Each element is a 4x3 numpy array, with the first 3 rows
  //!         forming the rotation, and the last row is the translation.
  //!         The affine transform is applied as rotation * vector + translation.
  //! \warning Works for primitive lattices only.
  //! \see Taken from Enum code, PRB 77, 224115 (2008).
  PYLADA_INLINE PyObject* space_group( Structure const &_lattice, 
                                     types::t_real _tolerance = -1e0 )
    PYLADA_END(return (*(PyObject*(*)(Structure const &, types::t_real))
                     api_capsule[PYLADA_SLOT(crystal)])(_lattice, _tolerance);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)space_group;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)


#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Creates list of first neigbors up to given input.
  //! \details Always return all nth neighbors. In other words, in fcc, if you
  //!          ask for 6 neighbor, actually 12 are returned. 
  //! \returns A python list of 3-tuples. Each tuple consists of a (unwrapped
  //!          python) reference to an atom, a vector which goes from the
  //!          center to the relevant periodic image of that neighbor, and the
  //!          distance between the center and that neighbor.
  PYLADA_INLINE PyObject* neighbors( Structure const &_structure,
                                   Py_ssize_t _nmax, 
                                   math::rVector3d const &_center,
                                   types::t_real _tolerance=types::tolerance )
    PYLADA_END(return (*(PyObject*(*)( Structure const&, 
                                     Py_ssize_t, 
                                     math::rVector3d const&, 
                                     types::t_real ))
                     api_capsule[PYLADA_SLOT(crystal)])
                    (_structure, _nmax, _center, _tolerance);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)neighbors;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Creates list of coordination shells up to given order.
  //! \returns A list of lists of tuples. The outer list is over coordination shells.
  //!          The inner list references the atoms in a shell.
  //!          Each innermost tuple contains a reference to the atom in question,
  //!          a translation vector to its periodic image inside the relevant shell, 
  //!          and the distance from the center to the relevant periodic image.
  //! \param[in] _structure : Structure for which to determine coordination shells.
  //! \param[in] _nshells : Number of shells to compute.
  //! \param[in] _center : Center of the coordination shells.
  //! \param[in] _tolerance : criteria to judge when a shell ends.
  //! \param[in] _natoms : Total number of neighbors to consider. Defaults to fcc + some security.
  PYLADA_INLINE PyObject* coordination_shells( crystal::Structure const &_structure,
                                             Py_ssize_t _nshells, 
                                             math::rVector3d const &_center,
                                             types::t_real _tolerance=types::tolerance,
                                             Py_ssize_t _natoms = 0 )
    PYLADA_END(return (*(PyObject*(*)( crystal::Structure const &,
                                     Py_ssize_t,
                                     math::rVector3d const &, 
                                     types::t_real, 
                                     Py_ssize_t ))
                     api_capsule[PYLADA_SLOT(crystal)])
                    (_structure, _nshells, _center, _tolerance, _natoms);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)coordination_shells;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)


#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Creates a split-configuration for a given structure and atomic origin.
  //! \details Split-configurations are a symmetry-agnostic atom-centered
  //!          description of chemical environment. For details, see
  //!          \url{http://dx.doi.org/10.1137/100805959} "d'Avezac, Botts,
  //!          Mohlenkamp, Zunger, SIAM J. Comput. \Bold{30} (2011)". 
  //! \param[in] _structure Structure for which to create split-configurations.
  //! \param[in] _index Index into _structure for the origin of the configuration.
  //! \param[in] _nmax Number of atoms (cutoff) to consider for inclusion
  //!                  in the split-configuration.
  //! \param[inout] _configurations Object where configurations should be
  //!       stored. It should a null object,  or a list of previously
  //!       existing configurations. There is no error checking, so do not
  //!       mix and match.
  //! \param[in] _tolerance Tolerance criteria when comparing distances.
  //! \return A list of splitted configuration. Each item in this list is
  //!         itself a list with two inner items. The first inner item is an
  //!         ordered list of references to atoms. The second inner item is
  //!         the weight for that configuration. The references to the atoms
  //!         are each a 3-tuple consisting of an actual reference to an
  //!         atom, a translation vector from the center of the configuration
  //!         to the atom's relevant periodic image, and a distance from the
  //!         center. [[[(atom, vector from center, distance from center),
  //!         ...], weight], ...]
  PYLADA_INLINE bool splitconfigs( Structure const &_structure,
                                 Atom const &_origin,
                                 Py_ssize_t _nmax,
                                 python::Object &_configurations,
                                 types::t_real _tolerance )
   PYLADA_END(return (*(bool(*)( Structure const&, Atom const&, 
                               Py_ssize_t, python::Object &,
                               types::t_real))
                    api_capsule[PYLADA_SLOT(crystal)])
                   (_structure, _origin, _nmax, _configurations, _tolerance);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)splitconfigs;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates divide and conquer box with periodic boundary condition.
  PYLADA_INLINE PyObject* dnc_boxes( const Structure &_structure, 
                                   math::iVector3d const &_mesh, 
                                   types::t_real _overlap )
  PYLADA_END(return (*(PyObject*(*)( Structure const&,
                                   math::iVector3d const&,
                                   types::t_real ))
                   api_capsule[PYLADA_SLOT(crystal)])
                  (_structure, _mesh, _overlap);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)dnc_boxes;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)
