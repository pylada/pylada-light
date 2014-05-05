.. _crystal_ug: 
.. currentmodule:: pylada.crystal

Creating and Manipulating crystal structures
********************************************

A good way to learn is to play with Pylada directly in the python interpreter.
Best of all, use the enhanced shell ipython_. It comes with many goodies. For
instance tab-completion. Or inline help. E.g., once you have typed up the first
example below, try ``structure.[TAB]`` and see what it tells you. Then try
``structure?`` to print the description (docstring_) of structure. If setup
right, you can even use it like an ordinary bash shell on `pythonic steroids`_.

Playing with the crystal structure
==================================

Initialization
--------------

To start off, lets create the diamond crystal structure.

>>> from pylada.crystal import Structure
>>> structure = Structure( [ [0, 0.5, 0.5],     \
...                          [0.5, 0, 0.5],     \
...                          [0.5, 0.5, 0] ] )
>>> structure.add_atom(0, 0, 0, "C")
>>> structure.add_atom(0.25, 0.25, 0.25, "C")

The first line above imports the :class:`Structure` class from the
:py:mod:`crystal <pylada.crystal>` module.  This class is the basic type which describes
crystal structures in Pylada.  The second and subsequent lines creates diamond.
The unit-cell is initialized within the first parenthesis. It must be given as
a *matrix*: the cell-vectors are columns (not rows as in many other physics
code). The reason behind choice will soon be apparent when we start playing the
unit-cell directly, entering it into mathematical equations as one would on
paper. At this point, we have a structure empty of any atoms. They can be added
as done above, first inserting the x, y, and z cartesian coordinates and then
the atomic occupation.

.. note:: Cartesian or fractional coordinates? Ask the question no more. Pylada
          *always* expects cartesian coordinates in real space. Period.
          However, keep reading to find an instance where the transformation is
          done. Just don't forget to transform back.

It is also possible to add whatever attributes directly when initializing the
structure. If one believes in d0 magnetization in carbon substituted by
Technetium, one could add a total moment to the structure, specify an atomic
site for Technetium substitutions, and markup another site with a spin
variable, all in a single one liner.

>>> from pylada.crystal import Structure
>>> structure = Structure( [ [0, 0.5, 0.5],                         \
...                          [0.5, 0, 0.5],                         \
...                          [0.5, 0.5, 0] ], scale=1.0, moment=5 ) \
...                      .add_atom(0, 0, 0, "C", spin="d0")         \
...                      .add_atom(0.25, 0.25, 0.25, "C", "Tc")

Note the backslash which tell the python interpreter to read everything as a
single line.

.. seealso::
   A number of predefined lattices can be found in
   :py:mod:`~pylada.crystal.bravais`, :py:mod:`~pylada.crystal.binary`,
   :py:mod:`~pylada.crystal.A2BX4`, :py:mod:`~pylada.crystal.ABX`. Other similar
   modules are always welcomed.

Manipulation
------------

So we've got a crystal structure. But how does one go around playing with it?
The cell can be accessed as follows:

>>> structure.cell[0, 1] = 0.6
>>> print structure.cell
[ [-0.5, 0.6, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5] ]
>>> structure.cell[1, :] = 0.1
>>> print structure.cell
[ [-0.5, 0.6, 0.5], [0.1, 0.1, 0.1], [0.5, 0.5, -0.5] ]

The first line in the code above accesses the x coordinate of the second cell
vector and changes it to 0.6. The second line of code modifies the y
coordinates of all three cell vectors simultaneously. Note that the cell a
numpy_ array. Numpy_ is python's numerical computation package. It does
everything that BLAS or Lapack does, but is much easier to use. And since, when
compiled correctly, it actually uses BLAS or similar library, it can be quite
fast. Most, perhaps all, arrays of numbers in Pylada are numpy_ arrays. 

Atoms in the structure can be accessed in the structure as though it were a list:

>>> atom0 = structure[0]
>>> atom1 = structure[1]
>>> atom0 is structure[-1]
True

Just like any list, the structure can be accessed starting from the end, using
negative integers. It is possible to loop over it as well:

>>> for atom in structure: print atom.pos
[0 0 0]
[0.25 0.25 0.25]

The above loops over all atoms and prints the position of each. Note that the
position is also a numpy_ array. Finally, the structure can be sliced:

>>> structure.add_atom(0.5, 0.5, 0.5, 'Pu')
>>> for atom in structure[1:]: print atom.type
['C' 'Tc']
'Pu'

The above adds a third atom to the structure. It then proceeds to loop over the
atoms, but skipping the first one. Remember we initialized the second atom with
two species? As such, it is actually a list. The third atom is actually just a
string:

>>> isinstance(structure[1].type, list)
True
>>> isinstance(structure[2].type, str)
True

You can set the type of an atom to anything you want. 
An integer:

>>> atom0.type = 0
>>> print structure[0].type
0

A boolean

>>> atom0.type = False
>>> print structure[0].type
False

Or even itself

>>> atom0.type = atom0
>>> print structure[0].type
Atom(0, 0, 0, 'C', spin='d0')

Though how that would be useful is not clear. The position, however, is
*always* a numpy_ array. Try otherwise, and you will get an error. Note above
that we set the type using ``atom0`` and then print it out using
``structure[0]``. ``atom0`` is a variable created earlier. It references the
structure's first atom. You can use one or the other. Both refer to the same
underlying atom.

The attribute ``spin`` can be accessed directly:

>>> print atom0.spin
"d0"
>>> atom0.foo = 'bar'

And new attributes can be added easily, whenever the need arises. The same goes
for ``moment`` defined in ``structure``'s initialization. Finally, lets do
some math:

>>> from numpy import dot
>>> from numpy.linalg import inv
>>> inverse_cell = inv(structure.cell)
>>> frac = dot(inverse_cell, structure[0].pos)

The snippet above computes the fractional coordinates of the first atomic
position. inv_ is a numpy_ method to invert 2d-arrays. dot_ provides
matrix-matrix multiplication and matrix-vector multiplications, and
vector-vector inner products.

.. note:: Why use dot_? By default, ``vectorA * vectorB`` multiplies arrays
          element per element in numpy_. There does exist a class called
          Matrix which will change the behavior to actual matrix
          mutliplication. But to avoid any surprises, Pylada uses the default.
          And hence makes use of dot_.


Specifiying units
-----------------

The units are given using the structure's scale attribute:

>>> from quantities import nanometer
>>> structure.scale = 0.5 * nanometer
>>> print structure.scale
5

The package quantities_ allows us to specify units explicitly. Note however
that the scale is converted to angstroms. Units are arbitrary and an arbitrary
choice was made to use angstroms throughout Pylada. quantities_ makes it
possible to convert back and forth between other preferred unit systems.
In other words, the unit-cell ``structure.scale * structure.cell`` is in
angstrom once multiplied by the scale. And so are the atomic positions
``structure.scale * structure[0].pos``.

More advanced structure manipulation methods
============================================

Supercells and primitive unit cells
-----------------------------------

Quite often, one needs to a supercell, i.e. a multiple, of a smaller unit cell.
Doing this takes all of a single line. 

>>> from pylada.crystal import Structure, supercell
>>> # First create unit cell.
>>> structure = Structure( [ [0, 0.5, 0.5],                  \
...                          [0.5, 0, 0.5],                  \
...                          [0.5, 0.5, 0] ] )               \
...                      .add_atom(0, 0, 0, "C")             \
...                      .add_atom(0.25, 0.25, 0.25, "C")
>>> # Now create conventional cell.
>>> conventional = supercell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], structure)
>>> conventional
Structure( 1, 0, 0,\
           0, 1, 0,\
           0, 0, 1, scale=1.0 )\
  .add_atom(0, 0, 0, 'C')\
  .add_atom(0.25, 0.25, 0.25, 'C') )\
  .add_atom(0, 0, 0, 'C', site=0)\
  .add_atom(0.25, 0.25, 0.25, 'C', site=1)\
  .add_atom(0.5, 0, 0.5, 'C', site=0)\
  .add_atom(0.75, 0.25, 0.75, 'C', site=1)\
  .add_atom(0.5, 0.5, 0, 'C', site=0)\
  .add_atom(0.75, 0.75, 0.25, 'C', site=1)\
  .add_atom(0, 0.5, 0.5, 'C', site=0)\
  .add_atom(0.25, 0.75, 0.75, 'C', site=1)

The above creates the conventional cell from the diamond unit cell. Note that
the new cell is given in cartesian coordinates (and in the units of the original
unit structure). Using dot_ and inv_, one could of course specify the
conventional cell in fractional coordinates:

>>> conventional = supercell( dot([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], inv(structure.cell)), structure)

It is just a bit more verbose.

Once a supercell is obtained, it is possible to go back to the original primitive unit-cell. 

>>> from pylada.crystal import primitive
>>> primitive(conventional, tolerance=1e-8)
Structure( 0.5, 0.5, 0,\
           0, 0.5, 0.5,\
           0.5, 0, 0.5,\
           scale=1 )\
  .add_atom(0, 0, 0, 'C')\
  .add_atom(0.25, 0.25, 0.25, 'C')

Note however that this method is far from perfect and is likely not robust with
respect to numerical noise. The optional ``tolerance`` keyword argument may
help to some degree. It defaults to 1e-8. There is also a method to check
whether a structure is indeed primitive.


>>> from pylada.crystal import primitive, is_primitive
>>> is_primitive(conventional)
False
>>> is_primitive(primitive(conventional))
True

It is also possible to obtain the cyclic groups of a supercell with respect to
its backbone lattice. In practice, it gives us a way to label an atom with
respect to the group of periodic images it belongs to [HF]_. 

>>> from random import randint
>>> from numpy import array, all, abs
>>> from pylada.crystal import SmithTransform
>>> # create smith transform.
>>> st = SmithTransform(conventional)
>>> # loop over random periodic images
>>> for i in xrange(10):
>>>   #
>>>   # create a vector with respect to its corresponding lattice site.
>>>   pos  = conventional[2].pos - lattice[ conventional[2].site ].pos
>>>   # 
>>>>  # random periodic image
>>>   translation = array([randint(-20, 20), randint(-20, 20), randint(-20, 20)], dtype="float64")
>>>   pos += dot(conventional.cell, translation)
>>>   # get indices in cyclic group.
>>>   indices = st.indices(pos)
>>>   # yep, its the same as the original atom.
>>>   assert all( abs( indices - [0, 0, 1] ) )

We check that statement in the code above. An atomic site is shifted to random
periodic images. Yet its cyclic indices remain the same. Note that Diamond has
two sublattices. This means it has two cyclic group [HF2]_, one for each atomic site
in the lattice. This is why the input to :py:func:`indices
<SmithTransform.indices>` is a vector which has been
translated back to the origin of its sublattice. We use the ``site`` attribute
added to each atom by :py:func:`supercell` to make the translation. This
attribute can also be set using the :py:func:`map_sites` method. 

.. seealso:: :py:func:`supercell`, :py:func:`primitive`, :py:func:`map_sites`,
             :py:func:`is_primitive`, :py:class:`SmithTransform`


Space-group operations
----------------------

The space group operations of a structure can also easily be obtained. This is
the operations, not the name of the space group. The return is a list of 4x3
numpy_ arrays where the upper 3x3 block is a rotation and the lowest row is a
translation. The translation should be applied *after* the rotation. Going back
to the original diamond structure:

>>> sg = space_group(structure)
>>> sg == space_group(structure) 
>>> len(sg)
48
>>> sg[0]
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  0.]])
>>> #
>>> # applying the operation:
>>> from numpy import dot, array
>>> dot(sg[1][:3], array([1., 1., 0])) + sg[1][3]
array([ 1., -1.,  0.])

A structure can easily be transform according to any affine transformation defined the same way:

>>> from pylada.crystal import transform
>>> transform(structure, sg[1])
Structure( 0, 0.5, 0.5,\
           -0.5, 0, -0.5,\
           -0.5, -0.5, 0,\
           scale=1 )\
  .add_atom(0, 0, 0, 'C')\
  .add_atom(0.25, -0.25, -0.25, 'C')

Of course, in this case the result is nothing more than a different
parameterization of the same lattice.

.. seealso:: :py:func:`space_group`, :py:func:`transform`

Neighbors and Coordination Shells
---------------------------------

A list of first neighbors can be obtained for any point in the structure.
Still using the diamond structure.

>>> from pylada.crystal import neighbors
>>> n = neighbors(structure, 5, [0.125,0.125, 0.125])
>>> len(n)
8

The first argument of ``neighbors`` is the structure, the second is the number
of neighbors to look for, and the third is the position for which to look for
neighbors. Note that 5 neighbors are requested, but 8 are actually returned.
The method always makes sure to return a complete coordination shell.
Coordinations are judged according to the distance from the central point. An
optional ``tolerance`` keyword argument exists defining how *equal* distances
are judged.

>>> n[0]
(Atom(0, 0, 0, 'C'), array([-0.125, -0.125, -0.125]), 0.21650635094610965)

Each item in the list returned by neighbor is a tuple consisting of reference
to the neighboring atom, a vector going from the central point to relevant
periodic image of that atom, and the distance from the center to the periodic
image. Additionally, a ``coordination_shells`` method exists which returns a
list of lists of neighbour, where each inner list is a single coordination
shell.

>>> from pylada.crystal import coordination_shell
>>> len( coordination_shell(structure, 5, [0.125, 0.125, 0.125])[0] )
2
>>> len( coordination_shell(structure, 5, [0.125, 0.125, 0.125])[1] )
6

.. seealso:: :py:func:`neighbors`, :py:func:`coordination_shells`

Input, output, saving to a file, sending as MPI message
-------------------------------------------------------

It is of course possible to save a structure to file:

>>> with open('text', 'w') as file: file.write(structure)

And do to read it out.

>>> with open('text', 'r') as file: structure = eval(file.read())

Note that structures and atoms are printed out as strings which can be executed
to retrieve the actual python object (i.e. it is representation_). The only
caveat is that the attributes you have added to the structure must also be
representable.

Structures and atoms can be pickled. pickle_ is a python module for data
retention. It transforms objects into a stream of characters which can be saved
to disk, sent as MPI messages, or whatever suits your fancy, and then
reinterpreted to become the same python objects all over again.
As far as MPI is concerned, however, the best bet is to use boost.mpi_. This
is a great python wrapper around the original MPI specifications. It truly makes
MPI easy.

Conclusion
----------

Pylada makes it easy to manipulate crystal structures any way you fancy. Further
methods exist beyond those described here. There is a method to create periodic
divide and conquer boxes, very practical when dealing with truly large
structures. There is a method to transform structures into lattice agnostic
representations, and others still to iterate over equivalent lattice sites,
atoms of nanowire shells. How the latter method was implemented is given as a
more advanced example below. Peek at it, and you will see how a fairly complex
functionality can be designed with only a few lines of codes. For the others,
however, please to the code itself, or to the API documentation.


Example: Iterating over the shells of a core-shell nanowire
===========================================================

Core shell nanowires are nano-structures where a thin nanowire of, say,
germanium, is coated with alternating layers of silicon and germanium. I will
now show how a few lines of code creates all the building blocks needed to look
at any possible arrangement of core-shells. This particular piece of code was
used to optimize light absorption at the band edges of Si/Ge core-shell
nanowires [ZALZ]_. 

The point here is to create an generator_ which will allow us to iterate over
shells in an outer loop and atoms (within the shell) in an inner loop. For
instance, if we wanted to alternate Si and Ge layers.

>>> from pylada.crystal import shell_iterator
>>> for i, shell in enumerate(shell_iterator(structure, center=[0,0,0], direction=[1, 0, 0])):
>>>    for atom in shell:
>>>      if i > 10: atom.type = 'Hg'
>>>      elif i % 2 == 0: atom.type = 'Si'
>>>      elif i % 2 == 1: atom.type = 'Ge'

The nanowire is created within a structure. In this case, it is likely a fairly
large supercell of zinc blende. Within it we want to place a nanowire with
given growth direction and a given center (the center can be on an atom, on a
bond, or somewhere else). The nanowire consists of 10 alternating layers of Si
and Ge, capped by a fake atom (in this case Hg). It is constructed using two
nested loops. The outer loop runs over shells, and the inner loop over atoms in
a shell. The enumerate_ method is a python primitive which counts the number
of iterations in a loop. It conveniently keeps track of which shell we are in,
so that we can alternate Si and Ge.

Now follows the code for the shell generator_.

>>> def shell_iterator(structure, center, direction, thickness=0.05):
>>>   """ Iterates over cylindrical shells of atoms.
>>>   
>>>       It allows to rapidly create core-shell nanowires.
>>>   
>>>       :Parameters:
>>>         structure : :class:`pylada.crystal.Structure`
>>>           Structure or Lattice over which to iterate.
>>>         center : 3d vector
>>>           Growth direction of the nanowire.
>>>         thickness : float
>>>           Thickness in units of ``structure.scale`` of an individual shell.
>>>       
>>>       :returns: Yields iterators over atoms in a single shell.
>>>   """
>>>   from operator import itemgetter
>>>   from numpy import array, dot
>>>   from numpy.linalg import norm
>>>
>>>   direction = array(direction)/norm(array(direction))
>>>   if len(structure) <= 1: yield structure; return
>>>
>>>   # orders position with respect to cylindrical coordinate.
>>>   positions = into_voronoi(array([atom.pos - center for atom in structure]), structure.cell)
>>>   projs = [(i, norm(pos - dot(pos, direction)*direction)) for i, pos in enumerate(positions)]
>>>   projs = sorted(projs, key=itemgetter(1))
>>>
>>>   # creates classes of positions.
>>>   result = {}
>>>   for i, proj in projs:
>>>     index = int(proj/thickness+1e-12)
>>>     if index in result: result[index].append(i)
>>>     else: result[index] = [i]
>>>
>>>   for key, layer in sorted(result.iteritems(), key=itemgetter(0)):
>>>     def inner_layer_iterator():
>>>       """ Iterates over atoms in a single layer. """
>>>       for index in layer: yield structure[index]
>>>     yield inner_layer_iterator()

As you can see, about a third of the code is comments.

>>>   direction = array(direction)/norm(array(direction))
>>>   if len(structure) <= 1: yield structure; return

The first line makes sure that the direction is normalized and is a numpy_
array. The second makes sure the structure is not too absurd.

>>>   # orders position with respect to cylindrical coordinate.
>>>   positions = into_voronoi(array([atom.pos - center for atom in structure]), structure.cell)
>>>   projs = [(i, norm(pos - dot(pos, direction)*direction)) for i, pos in enumerate(positions)]
>>>   projs = sorted(projs, key=itemgetter(1))

The crux are the three lines of code above. Basically, we want to transform the
atoms from cartesian to cylindrical coordinates. However, there is a trick. The
structure is periodic and we have first to make sure that we are looking at the
periodic images which are closest to the center. That is the function of the
``into_voronoi`` method. It takes a vector (or array of vectors) and a cell
matrix, and folds the former into the Wigner-Seitz cell (aka first brillouin
zone, aka first Voronoi region of ``center``). The second line creates a list
of tuples, where the first item is an index into the structure, and the second
item is the cylindrical coordinate ``r``. Now, all we need do is sort the list
with respect to ``r``. This is the third line. getitem_ tells python to sort
with respect to the second item in each tuple. 

.. note:: Technically, we should rather find the periodic image with the
          smallest ``r`` component. However, since the supercell is (generally)
          much smaller in the ``z`` direction, it has yet never mattered.


We now have a sorted list of cylindrical coordinates to which are attached the
index of each corresponding atom in the structure. At this point, we need to
create a mapping from the index of the shell to the relevant items in the list
we just created. Note that the thickness of the shell is actually an external
parameter.

>>>   # creates classes of positions.
>>>   result = {}
>>>   for i, r in projs:
>>>     index = int(r/thickness+1e-12)
>>>     if index in result: result[index].append(i)
>>>     else: result[index] = [i]

Finally, the rest of the code is concerned with making the outer and inner
loops possible. 

>>> for key, layer in sorted(result.iteritems(), key=itemgetter(0)):
>>>   def inner_layer_iterator():
>>>     """ Iterates over atoms in a single layer. """
>>>     for index in layer: yield structure[index]
>>>   yield inner_layer_iterator()

We loop over the (sorted) keys in the mapping just created. Each iteration
visits a different shell, starting with the innermost. For each, we yield_ a
generator_ which will loop over the atoms in the shell. To yield means we
return *temporarily* from ``shell_iterator``. However, the next time the user
asks to loop to the next shell, the python interpreter knows to reenter
``shell_iterator`` *right after the yield statement*, as though it had never
left.  I strongly recommend taking a closer look at this yield_ deal.
Programming is often about doing loops and yield_, once you've warped your
brain around the concept, makes it extremely easy.

.. seealso:: :class:`shell_iterator`, :class:`layer_iterator`, :class:`equivalence_iterator`

.. _docstring: http://en.wikipedia.org/wiki/Docstring#Python
.. _pythonic steroids: http://ipython.org/ipython-doc/rel-0.12/interactive/tutorial.html#system-shell-commands
.. _inv: http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
.. _dot: http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
.. _representation: http://docs.python.org/library/functions.html#repr
.. _pickle: http://docs.python.org/library/pickle.html 
.. _boost.mpi: http://www.boost.org/doc/libs/1_35_0/doc/html/mpi/python.html
.. _generator: http://docs.python.org/tutorial/classes.html#generators
.. _yield: http://docs.python.org/tutorial/classes.html#generators
.. _enumerate: http://docs.python.org/library/functions.html?highlight=enumerate#enumerate
.. _getitem: http://docs.python.org/library/operator.html#operator.__getitem__

.. [HF] Gus L. Hart, Rodney W. Forcade,
        `Algorithm for generating derivative structures`,
        Phys. Rev. B **77**, 224115 (2008),
        http://dx.doi.org/10.1103/PhysRevB.77.224115
.. [HF2] Gus L. Hart, Rodney W. Forcade,
         `Generating derivative structures from multilattices: Algorithm and
         application to hcp alloys`,
         Phys. Rev. B **80**, 014120 (2009),
         http://dx.doi.org/10.1103/PhysRevB.80.014120
.. [ZALZ] Lijun Zhang, Mayeul d'Avezac, Jun-Wei Luo, Alex Zunger 
          `Genomic Design of Strong Direct-Gap Optical Transition in Si/Ge
          Core/Multishell Nanowires`,
          Nano Lett. **12**, 984-991 (2012),
          http://dx.doi.org/10.1021/nl2040892
