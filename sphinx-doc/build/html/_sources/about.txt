About
=====

Pylada is a modular python framework to control physics simulations, from DFT to
empirical pseudo-potentials to point ion electrostatics. It's goal is to
provide the basic building blocks from which methods incorporating different
Hamiltonians can be constructed. It is designed around three main concepts:

  * constructing and manipulating periodic crystal structures. For instance, it
    is possible, starting from the unit cell of spinel to automatically create
    a supercell with vacancies or substitutions.
  * manipulating functionals, such as VASP, and extracting their output. For
    instance, one could throw diamond at VASP, expect it to relax the
    structure, then perform a static calculation for maximum accuracy. Or
    possibly perform an epitaxial relaxation, something VASP does not do
    per say.
  * Manipulate, launch, and check the results for thousands of calculations
    simultaneously. Think of calculating different of carbone allotropes. It
    would be impractical to qsub each and every job and then check that each
    ran correctly. Hence, Pylada provides an interface for that.

With these building blocks in hand, it is possible, fairly easy, and - I'm told
- even fun,  to construct complex computational routines. Computing phonons or
formation enthalpy requires some know-how, certainly. But it is also quite
repetitive. Pylada takes the gruelling out of it.

Here are a few of the projects Pylada helped us with.

  * *When brute force suffices:* A\ :sub:`2`\ BX\ :sub:`4` (A, B cations, X=O, S,
    Se, Te) form a vast family of compounds crystallizing in forty different
    structures. However, hundreds of possible combinations of A, B, X have
    never been identified in nature before. Do these exist and have simply
    never been reported? Or are they thermodynamically unstable and could never
    have been found? To answer this question, we systematically computed the
    formation enthalpy of hundreds of unknown A\ :sub:`2`\ BX\ :sub:`4` as well
    as that of their (reported) competing binaries and ternaries. We found
    indeed quite a few compounds, some which may be fairly easily grown, which
    simply passed through the sieve. Pylada made it a sinch to perform these
    individual calculations. It allowed us to shove them in a database and then
    scour the data to automatically create phase diagrams for each system. 

  * *Stay abstract and leave off mano a mano calculations:* One of the goals of
    the project described above was to explore stable A\ :sub:`2`\ BX\ :sub:`4`
    for semi-conducting properties. More specifically, we were interested in
    predicting whether these compounds can be doped and how much. Or whether
    some point defects intrinsic to the material would form spontaneously when
    external dopants are introduced and limit the material's carrier
    concentration. To do this, we created a simple script to automatically
    determine possible vacancies and substitutions, as well as their charge
    state. These defects are all `the same`__, so no need to worry and create
    each, one at a time.  Pylada made it possible to create the systems, launch
    the appropriate chain of calculations, and gather the results, and let
    physicists do physics rather than word processing input files.
 
  * *Manipulating DFT functionals:* `This DFT functional`__ was introduced to
    reproduce GW band-structures. It works by fitting a non-local potential.
    Originally, each fit would be performed by hand for each element. It turned
    out to be tedious. Especially since for more complicated compounds one
    would rather fit all the elements involved simultaneously. But Pylada offers
    python wrappers around DFT codes, and python offers a number of
    optimization functions. The two together make it possible to carry out
    painless optimization of a empirical DFT functional.

  * *Searching for and finding a needle:* Silicon is a possibly God's greatest gift
    for nerds everywhere. It is n-type. It is p-type. It naturally forms a
    protective insulating oxide layer. However, it is an indirect gap material
    and absorbs visible light only through the mediation of phonons. It has
    been known for a while that it is possible to create Si/Ge superlattices
    which are nominally direct gap. Indeed, Si\ :sub:`6`\ Ge\ :sub:`4` on a
    strained (001) substrate absorb light at the band edges. But only
    academically so. It is possible that a superlattice with specific motif and
    Si and Ge layer with meaningful photon absorption exists somewhere, but
    there a literally billions upon trillions of possibilities. So which one to
    start off with? At that point, research tapered off. We combined an
    empirical pseudo-potential method with a genetic algorithm to perform the
    search for us. Eventually, the search `payed off`__.

  * *screwing up:* Speaking from painful and embarrassing personal experience,
    it is extremely easy to launch tens and hundreds of meaningless jobs. The
    physics is still for the user to figure out. 

.. __: http://dx.doi.org/10.1088/0965-0393/17/8/084002
.. __: http://dx.doi.org/10.1103/PhysRevB.77.241201
.. __: http://dx.doi.org/10.1103/PhysRevLett.108.027401
