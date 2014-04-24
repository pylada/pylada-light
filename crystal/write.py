###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
# 
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
#  crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
# 
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
# 
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
# 
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################

""" Methods to write structures from file. """
def poscar(structure, file='POSCAR', vasp5=None, substitute=None):
  """ Writes a poscar to file. 

      :param structure:
          The structure to print out.
      :type structure:
          :py:class:`Structure`
      :param file:
          Object with a ''write'' method. If a string, then a file object is
          opened with that filename. The file is overwritten. If None, then
          writes to POSCAR in current working directory.
      :type file: str, stream, or None.
      :param bool vasp5:
          If true, include species in poscar, vasp-5 style.  Otherwise, looks
          for :py:data:`is_vasp_4 <pylada.is_vasp_4>` global config variable. 
          Defaults to False, in which case, does not print specie types.
      :param substitute:
          If present, will substitute the atom type in the structure. Can be
          incomplete. Only works with vasp5 = True (or :py:data:`is_vasp_4
          <pylada.is_vasp_4>` = True).
      :type substitute:
          dict or None
  
      >>> with open("POSCAR", "w") as file: write.poscar(structure, file, vasp5=True)

      Species in structures can be substituted for others (when using vasp5 format).
      Below, aluminum atoms are replaced by cadmium atoms. Other atoms are left unchanged.

      >>> with open("POSCAR", "w") as file:
      >>>   write.poscar(structure, file, vasp5=True, substitute={"Al":"Cd"})

      Selective dynamics are added to the POSCAR file if an atom in the
      structure has a freeze attribute (of non-zero length). It is expected
      that this attribute is a string and that it contains one of "x", "y",
      "z", corresponding to freezing the first, second, or third fractional
      coordinates. Combinations of these are also allowed.
  """
  from quantities import angstrom
  if file is None:
    with open('POSCAR', 'w') as fileobj: return poscar(structure, fileobj, vasp5, substitute)
  elif not hasattr(file, 'write'):
    with open(file, 'w') as fileobj: return poscar(structure, fileobj, vasp5, substitute)

  from numpy import matrix, dot
  from . import specieset

  if vasp5 is None:
    import pylada 
    vasp5 = not getattr(pylada, 'is_vasp_4', True)

  string = "{0}\n{1}\n".format(getattr(structure, 'name', ''),
                               float(structure.scale.rescale(angstrom)))
  for i in range(3):
    string += "  {0[0]} {0[1]} {0[2]}\n".format(structure.cell[:,i])
  species = specieset(structure)
  if vasp5: 
    if substitute is None: substitute = {}
    for s in species: string += " {0} ".format(substitute.get(s,s))
    string += "\n"
  for s in species: 
    string += "{0} ".format(len([0 for atom in structure if atom.type == s]))
  inv_cell = matrix(structure.cell).I
  selective_dynamics =\
      any([len(getattr(atom, 'freeze', '')) != 0 for atom in structure])
  if selective_dynamics: string += "\nselective dynamics\ndirect\n"
  else: string += '\ndirect\n'
  for s in species: 
    for atom in structure:
      if atom.type != s: continue
      string += "  {0[0]} {0[1]} {0[2]}"\
                .format(dot(inv_cell, atom.pos).tolist()[0])
      freeze = getattr(atom, 'freeze', '')
      if selective_dynamics:
        string += "  {1} {2} {3}\n"\
                    .format( 'T' if 'x' in freeze  != 0 else 'F', 
                             'T' if 'y' in freeze  != 0 else 'F', 
                             'T' if 'z' in freeze  != 0 else 'F' ) 
      else: string += '\n'
  if file == None: return string
  elif isinstance(file, str): 
    from ..misc import RelativePath
    with open(RelativePath(file).path, 'w') as file: file.write(string)
  else: file.write(string)

def castep(structure, file=None):
  """ Writes castep input. """
  from quantities import angstrom
  cell = structure.cell * float(structure.scale.rescale(angstrom))
  string = "%BLOCK LATTICE_CART\n" \
           "  {0[0]} {0[1]} {0[2]}\n" \
           "  {1[0]} {1[1]} {1[2]}\n" \
           "  {2[0]} {2[1]} {2[2]}\n" \
           "%ENDBLOCK LATTICE_CART\n\n"\
           "%BLOCK POSITIONS_ABS\n".format(*(cell.T))
  for atom in structure:
    pos = atom.pos * float(structure.scale.rescale(angstrom))
    string += "  {0} {1[0]} {1[1]} {1[2]}\n"\
              .format(atom.type, pos)
  string += "%ENDBLOCK POSITION_ABS\n"
  if file == None: return string
  elif isinstance(file, str): 
    from ..misc import RelativePath
    with open(RelativePath(file).path, 'w') as file: file.write(string)
  else: file.write(string)

def crystal( structure, file='fort.34',
             dimensionality=None, centering=None, type=None, spacegroup=None ):
  """ Writes structure as CRYSTAL's EXTPRT. 

      :param structure:
          The structure to print out.
      :type structure: :py:class:`Structure`
      :param file:
          Object with a ''write'' method. If a string, then a file object is
          opened with that filename. The file is overwritten. If None, then
          returns a string.
      :type file: str, stream, or None.
      :param int dimensionality:
          Dimensionality of the system as an integer between 0 and  3 included.
          If None, checks for a ''dimensionality'' attribute in ``structure``.
          If that does not fit the biil, defaults to 3.
      :param int centering:
          Centering in CRYSTAL_'s integer format. Is None, checks ``structure``
          for a ``centering`` integer attribute. If that does not exist or is
          not convertible to an integer, then defaults to 1.
      :param int type:
          Crystal type in CRYSTAL_'s integer format. Is None, checks
          ``structure`` for a ``type`` integer attribute. If that does not
          exist or is not convertible to an integer, then defaults to 1.
      :param spacegroup:
          The structure's space group as a sequence of 4x3 matrices. If this is
          None (default), then checks for ''spacegroup'' attributes. If that
          does not exist, uses :py:function:`~pylada.crystal.space_group`.
  """
  from StringIO import StringIO
  from numpy import zeros
  from quantities import angstrom
  from ..periodic_table import find as find_specie
  from .iterator import equivalence as equivalence_iterator
  from . import space_group
  # makes sure file is a stream.
  # return string when necessary
  if file is None:
    file = StringIO()
    crystal(structure, file, dimensionality, centering, type, spacegroup)
    return file.getvalue()
  elif not hasattr(file, 'write'):
    with open(file, 'w') as fileobj:
      return crystal( structure, fileobj, dimensionality,
                      centering, type, spacegroup)
  
  # normalize input as keyword vs from structure vs default.
  try:
    if dimensionality is None:
      dimensionality = getattr(structure, 'dimensionality', 3)
    dimensionality = int(dimensionality)
    if dimensionality < 0 or dimensionality > 3: dimensionality = 3
  except: dimensionality = 3
  try:
    if centering is None: centering = getattr(structure, 'centering', 1)
    centering = int(centering)
  except: centering = 1
  try:
    if type is None: type = getattr(structure, 'type', 1)
    type = int(type)
  except: type = 1
  if spacegroup is None: spacegroup = getattr(structure, 'spacegroup', None)
  if spacegroup is None: spacegroup = space_group(structure)
  if len(spacegroup) == 0:
    spacegroup = zeros((1, 4, 3))
    spacegroup[0,0,0] = 1
    spacegroup[0,1,1] = 1
    spacegroup[0,2,2] = 1

  # write first line
  file.write('{0} {1} {2}\n'.format(dimensionality, centering, type))
  # write cell
  cell = structure.cell * float(structure.scale.rescale(angstrom))
  file.write( '{0[0]: > 18.12f} {0[1]: > 18.12f} {0[2]: > 18.12f}\n'           \
              '{1[0]: > 18.12f} {1[1]: > 18.12f} {1[2]: > 18.12f}\n'           \
              '{2[0]: > 18.12f} {2[1]: > 18.12f} {2[2]: > 18.12f}\n'           \
              .format( *(cell) ) )
  # write symmetry operators
  file.write('{0}\n'.format(len(spacegroup)))
  for op in spacegroup:
    file.write( '{0[0]: > 18.12f} {0[1]: > 18.12f} {0[2]: > 18.12f}\n'         \
                '{1[0]: > 18.12f} {1[1]: > 18.12f} {1[2]: > 18.12f}\n'         \
                '{2[0]: > 18.12f} {2[1]: > 18.12f} {2[2]: > 18.12f}\n'         \
                .format(*op[:3]) )
    file.write( '    {0[0]: > 18.12f} {0[1]: > 18.12f} {0[2]: > 18.12f}\n'     \
                .format(op[3]) )


  # figure out inequivalent atoms.
  groups = [u for u in equivalence_iterator(structure, spacegroup)]
  file.write('{0}\n'.format(len(groups)))
  for group in groups:
    atom = structure[group[0]]
    # Try figuring out atomic number.
    type = atom.type
    try: n = int(type)
    except: 
      try: n = find_specie(name=type)
      except:
        raise ValueError( 'Could not transform {0} to atomic number.'          \
                          .format(type) )
      else: type = n.atomic_number
    else: type = n
    pos = atom.pos * float(structure.scale.rescale(angstrom))
    file.write( '{0: >5} {1[0]: > 18.12f} {1[1]: > 18.12f} {1[2]: > 18.12f}\n' \
                .format(type, pos) )


def gulp(structure, file='gulp.in', **kwargs):
  """ Writes structure as gulp input. 
  
      :param structure:
         A :py:class:`Structure` or
         :py:class:`~lada.dftcrystal.crystal.Crystal` instance to write in GULP
         format.
      :param file:
         If a string, the it should be a filename. If None, then this function
         will return a string with structure in GULP input format. Otherwise,
         it should be a stream with a ``write`` method.
      :param name: 
         Name of the structure. Defaults to ``structure.name`` if the attribute
         exists. Does not include in input name otherwise.
      :param int periodicity:
         Dimensionality of the structure. Defaults to figuring it out from the
         input structure.
      :param int symmgroup:
         The name or number of the symmetry group. Defaults to
         ``structure.symmgroup`` or ``structure.spacegroup`` or 1. If it is not
         one, then it should be possible to figure out what the symmetry
         operators are. An error will be thrown otherwise.
      :params list freeze:
         List of 6 boolean specifying whether a strain degree of freedom is
         *not* allowed to relax. This is the opposite convention from GULP.
         The list determines each coordinate xx, yy, zz, yz, xz, xy.

      Only the asymmetric (a.k.a symmetrically inequivalent atoms) should be
      written to the file. As a result, we need a way to determine which atoms
      are asymmetric. There are a number of ways to do this:

        - the symmetry group (``symmgroup`` keyword) is 1. Hence all atoms are
          asymmetric
        - the input is a :py:class:`~lada.dftcrystal.crystal.Crystal` object,
          and hence everything about its symmetries is known
        - atoms are marked with a boolean attribute ``asymmetric``
        - a keyword argument or an attribute ``symmops`` or ``spacegroup``
          exist containing a list of symmetry operations. These should be all
          the operations needed to determine the asymmetric unit cells.
        - the symmetries are determined using :py:func:`space_group`. This is
          the last resort.

       Furthermore, each atom can have a freeze parameter 
  """
  from quantities import angstrom
  from .iterator import equivalence as equivalence_iterator
  from . import space_group, _normalize_freeze_cell, _normalize_freeze_atom

  def getvalue(name, defaults=None):
    """ returns value from kwargs or structure. """
    return kwargs.get(name, getattr(structure, name, defaults))

  result = ""
  # dump name or title
  name = getvalue('name')
  if name is None: name = getvalue('title')
  if name is not None: 
    name = name.rstrip().lstrip()
    if len(name) > 0: result += 'name\n{0}\n'.format(name)

  # Makes sure structure is evaluated (dftcrystal.Crystal) first if necessary.
  crystal = structure if not hasattr(structure, 'eval') else structure.eval()

  # figures out symmetry group
  symmgroup = getvalue('symmgroup')
  if symmgroup is None: symmgroup = getvalue('spacegroup', None)
  if symmgroup is None or hasattr(symmgroup, '__iter__'):
    symmgroup = getvalue('space')
  # figure out periodicity and dumps appropriate cell.
  periodicity = getvalue('periodicity')
  if periodicity is None:
    periodicity = 3
    if abs(crystal.cell[2, 2] - 500) < 1e-8: periodicity -= 1
    if abs(crystal.cell[1, 1] - 500) < 1e-8: periodicity -= 1
    if abs(crystal.cell[0, 0] - 500) < 1e-8: periodicity -= 1
  if periodicity == 3:
    result += 'vectors\n'                                                      \
              '{0[0]: <18.8f} {0[1]: <18.8f} {0[2]: <18.8f}\n'                 \
              '{1[0]: <18.8f} {1[1]: <18.8f} {1[2]: <18.8f}\n'                 \
              '{2[0]: <18.8f} {2[1]: <18.8f} {2[2]: <18.8f}\n'                 \
              .format(*crystal.cell.T)                                       
    # dump spacegroup number or string
    if symmgroup is not None: result += 'spacegroup\n{0}\n'.format(symmgroup)
    # freeze cell parameters
    freeze = getvalue('freeze')
    if freeze is not None:
      freeze = _normalize_freeze_cell(freeze)
      result += ' '.join(('0' if f else '1') for f in freeze) + '\n'
  elif periodicity == 2:                                                       
    result += 'svectors\n'                                                     \
              '{0[0]: <18.8f} {0[1]: <18.8f}\n'                                \
              '{1[0]: <18.8f} {1[1]: <18.8f}\n'                                \
              .format(*crystal.cell.T)                                         
    # freeze cell parameters
    freeze = getvalue('freeze')
    if freeze is not None:
      freeze = _normalize_freeze_cell(freeze, 2)
      result += ' '.join(('0' if f else '1') for f in freeze) + '\n'
  elif periodicity == 1: raise NotImplementedError('Cannot do 1d materials')
  elif periodicity == 0: raise NotImplementedError('Cannot do 0d materials')

  if len(crystal) == 0: return result
  charges = getvalue('charges', {})
  
  # figures out asymmetric atoms.
  asymatoms = crystal if symmgroup == 1 or symmgroup is None else None
  if asymatoms is None and all([hasattr(u, 'asymmetric') for u in crystal]):
    asymatoms = [u for u in crystal if u.asymmetric]
  if asymatoms is None:
    symmops = getvalue('symmops', None)
    if symmops is None: symmops = getvalue('spacegroup', None)
    if symmops is None and crystal is not structure                            \
       and hasattr(structure, 'symmetry_operators'):
      symmops = structure.symmetry_operators
    if symmops is None or not hasattr(symmops, '__iter__'):
      symmops = space_group(crystal)
    if symmops is None or not hasattr(symmops, '__iter__'): 
      raise ValueError('Could not determine symmetry operations')
    asymatoms = [crystal[u[0]] for u in equivalence_iterator(crystal, symmops)]

  # now dumps atoms into seperate regions, if need be. 
  regions = {}
  for atom in asymatoms:
    pos = atom.pos * crystal.scale.rescale(angstrom).magnitude
    string = '{0.type:<4} core {1[0]:> 18.8f} {1[1]:> 18.8f} {1[2]:> 18.8f} '  \
              .format(atom, pos)
    charge = getattr(atom, 'charge', charges.get('charge', None))
    if hasattr(charge, 'rescale'): charge = float(charge.rescale('e'))
    if charge is not None: string += str(charge)
    freeze = getattr(atom, 'freeze', None)
    if freeze is not None: 
      freeze = _normalize_freeze_atom(freeze)
      string += ' '.join(('0' if f else '1') for f in freeze)
    string += '\n'
    if getattr(atom, 'shell', False) is not False:
      pos = getattr(atom.shell, 'pos', atom.pos)
      pos = pos * crystal.scale.rescale(angstrom).magnitude
      string += '{1:<4} shell {0[0]:> 18.8f} {0[1]:> 18.8f} {0[2]:> 18.8f} '   \
                .format(pos, atom.type)
      freeze = getattr(atom.shell, 'freeze', freeze)
      if freeze is not None: 
        freeze = _normalize_freeze_atom(freeze)
        string += ' '.join(('0' if f else '1') for f in freeze)
      string += '\n'
    region = getattr(atom, 'region', 0)
    if region not in regions: regions[region] = string
    else: regions[region] += string

  # now dump regions to result
  region_keys = getvalue('regions', {})
  for key, value in regions.iteritems():
    keyword = getvalue('region{0}'.format(key), region_keys.get(region, ''))

    if key != 0: index = 'region {0}'.format(key)
    elif len(keyword) == 0: index = '' 
    else: index = 'region {0}'.format(max(len(regions) + 1, 3))

    result += 'cartesian {0} {1}\n{2}\n'.format(index, keyword, value)
    
  return result
