from pylada.crystal import supercell, Structure, read
#from lada.crystal import supercell, Structure, read
from numpy import arange, sqrt, array, transpose, pi, dot, cross, sum
from numpy.linalg import det, inv
#############################################


def make_surface(structure=None, miller=None, nlayers=5, vacuum=15, acc=5):
    """Returns a slab from the 3D structure 

       Takes a structure and makes a slab defined by the miller indices 
       with nlayers number of layers and vacuum defining the size 
       of the vacuum thickness. Variable acc determines the number of 
       loops used to get the direct lattice vectors perpendicular 
       and parallel to miller. For high index surfaces use larger acc value 
       .. warning: (1) cell is always set such that miller is alogn z-axes
                   (2) nlayers and vacuum are always along z-axes.

       :param structure: LaDa structure
       :param miller: 3x1 float64 array
           Miller indices defining the slab    
       :param nlayers: integer
           Number of layers in the slab
       :param vacuum: real
           Vacuum thicness in angstroms
       :param acc: integer
           number of loops for finding the cell vectors of the slab structure
    """
    direct_cell = transpose(structure.cell)
    reciprocal_cell = 2 * pi * transpose(inv(direct_cell))

    orthogonal = []  # lattice vectors orthogonal to miller

    for n1 in arange(-acc, acc + 1):
        for n2 in arange(-acc, acc + 1):
            for n3 in arange(-acc, acc + 1):

                pom = array([n1, n2, n3])
                if dot(pom, miller) == 0 and dot(pom, pom) != 0:
                    orthogonal.append(array([n1, n2, n3]))

    # chose the shortest parallel and set it to be a3 lattice vector
    norm_orthogonal = [sqrt(dot(dot(x, direct_cell), dot(x, direct_cell))) for x in orthogonal]
    a1 = orthogonal[norm_orthogonal.index(min(norm_orthogonal))]

    # chose the shortest orthogonal to miller and not colinear with a1 and set it as a2
    in_plane = []

    for x in orthogonal:
        if dot(x, x) > 1e-3:
            v = cross(dot(x, direct_cell), dot(a1, direct_cell))
            v = sqrt(dot(v, v))
            if v > 1e-3:
                in_plane.append(x)

    norm_in_plane = [sqrt(dot(dot(x, direct_cell), dot(x, direct_cell))) for x in in_plane]
    a2 = in_plane[norm_in_plane.index(min(norm_in_plane))]

    a1 = dot(a1, direct_cell)
    a2 = dot(a2, direct_cell)

    # new cartesian axes z-along miller, x-along a1, and y-to define the right-hand orientation
    e1 = a1 / sqrt(dot(a1, a1))
    e2 = a2 - dot(e1, a2) * e1
    e2 = e2 / sqrt(dot(e2, e2))
    e3 = cross(e1, e2)

    # find vectors parallel to miller and set the shortest to be a3
    parallel = []

    for n1 in arange(-acc, acc + 1):
        for n2 in arange(-acc, acc + 1):
            for n3 in arange(-acc, acc + 1):
                pom = dot(array([n1, n2, n3]), direct_cell)
                if sqrt(dot(pom, pom)) - dot(e3, pom) < 1e-8 and sqrt(dot(pom, pom)) > 1e-3:
                    parallel.append(pom)

    # if there are no lattice vectors parallel to miller
    if len(parallel) == 0:
        for n1 in arange(-acc, acc + 1):
            for n2 in arange(-acc, acc + 1):
                for n3 in arange(-acc, acc + 1):
                    pom = dot(array([n1, n2, n3]), direct_cell)
                    if dot(e3, pom) > 1e-3:
                        parallel.append(pom)

    parallel = [x for x in parallel if sqrt(
        dot(x - dot(e1, x) * e1 - dot(e2, x) * e2, x - dot(e1, x) * e1 - dot(e2, x) * e2)) > 1e-3]
    norm_parallel = [sqrt(dot(x, x)) for x in parallel]

    assert len(norm_parallel) != 0, "Increase acc, found no lattice vectors parallel to (hkl)"

    a3 = parallel[norm_parallel.index(min(norm_parallel))]

    # making a structure in the new unit cell - defined by the a1,a2,a3
    new_direct_cell = array([a1, a2, a3])

    assert abs(det(new_direct_cell)) > 1e-5, "Something is wrong your volume is equal to zero"

    # make sure determinant is positive
    if det(new_direct_cell) < 0.:
        new_direct_cell = array([-a1, a2, a3])

    #structure = fill_structure(transpose(new_direct_cell),structure.to_lattice())
    structure = supercell(lattice=structure, supercell=transpose(new_direct_cell))

    # transformation matrix to new coordinates x' = dot(m,x)
    m = array([e1, e2, e3])

    # seting output structure
    out_structure = Structure()
    out_structure.scale = structure.scale
    out_structure.cell = transpose(dot(new_direct_cell, transpose(m)))

    for atom in structure:
        p = dot(m, atom.pos)
        out_structure.add_atom(p[0], p[1], p[2], atom.type)

    # repaeting to get nlayers and vacuum
    repeat_cell = dot(out_structure.cell, array([[1., 0., 0.], [0., 1., 0.], [0., 0., nlayers]]))
    out_structure = supercell(lattice=out_structure, supercell=repeat_cell)

    # checking whether there are atoms close to the cell faces and putting them back to zero
    for i in range(len(out_structure)):
        scaled_pos = dot(out_structure[i].pos, inv(transpose(out_structure.cell)))
        for j in range(3):
            if abs(scaled_pos[j] - 1.) < 1e-5:
                scaled_pos[j] = 0.
        out_structure[i].pos = dot(scaled_pos, transpose(out_structure.cell))

    # adding vaccum to the cell
    out_structure.cell = out_structure.cell + \
        array([[0., 0., 0.], [0., 0., 0.], [0., 0., float(vacuum) / float(out_structure.scale)]])

    # translating atoms so that center of the slab and the center of the cell along z-axes coincide
    max_z = max([x.pos[2] for x in out_structure])
    min_z = min([x.pos[2] for x in out_structure])
    center_atoms = 0.5 * (max_z + min_z)
    center_cell = 0.5 * out_structure.cell[2][2]

    for i in range(len(out_structure)):
        out_structure[i].pos = out_structure[i].pos + array([0., 0., center_cell - center_atoms])

    # exporting the final structure
    return out_structure

##########################################################################


def sort_under_coord(bulk=None, slab=None):
    """Returns indices and th coordinations of the undercoordinated atoms in a slab created from the bulk 

       :param bulk: pylada structure
       :param slab: pylada structure
    """
    from pylada.crystal import neighbors

    # Check the coordination in the bulk first shell
    bulk_first_shell = []
    for atom in bulk:
        bulk_first_shell.append(
            [atom.type, len(neighbors(structure=bulk, nmax=1, center=atom.pos, tolerance=1e-1 / bulk.scale))])

    del atom

    maxz = max([x.pos[2] for x in slab])
    minz = min([x.pos[2] for x in slab])

    # Find the undercoordinated atoms in the slab
    under_coord = []
    indices = [br for br in range(len(slab)) if slab[br].pos[
        2] <= minz + 4. / float(slab.scale) or maxz - 4. / float(slab.scale) <= slab[br].pos[2]]

    for i in indices:
        atom = slab[i]
        coordination = len(neighbors(structure=slab, nmax=1,
                                     center=atom.pos, tolerance=1e-1 / slab.scale))

        # Find the equivalent bulk atom to compare the coordination with
        for j in range(len(bulk)):
            eq_site = float(atom.site - j) / float(len(bulk))
            if abs(int(eq_site) - eq_site) < 1e-2:
                break

        # Comparing coordination with the "equivalent" bulk atom
        if coordination != bulk_first_shell[j][1]:
            under_coord.append([i, coordination])

        del coordination

    # returns the list of undercoordinated atoms,
    # atom index in the slab, coordination
    return under_coord

##########################################################################
# Miscellaneous


def shift_to_bottom(structure=None, atom=None, vacuum=None):
    shift_vector = transpose(structure.cell)[2] - array([0., 0., vacuum / structure.scale])
    structure[atom].pos = structure[atom].pos - shift_vector


def shift_to_top(structure=None, atom=None, vacuum=None):
    shift_vector = transpose(structure.cell)[2] - array([0., 0., vacuum / structure.scale])
    structure[atom].pos = structure[atom].pos + shift_vector


def z_center(slab=None):
    return sum(array([atom.pos[2] for atom in slab])) / len(slab)


def dipole_moment(slab=None, charge=None):
    """Claculates the dipole moment"""
    z = []
    ch = []
    for atom in slab:
        z.append(atom.pos[2])
        ch.append(charge[atom.type])
    assert not abs(sum(array(ch))) > 1e-5, "System is not charge neutral!"
    dipole_moment = sum(array(ch) * array(z))
    return dipole_moment
##########################################################################


def count_broken_bonds(bulk=None, slab=None):
    """Counts broken bonds per atom"""

    from pylada.crystal import neighbors

    under_coord = sort_under_coord(bulk=bulk, slab=slab)
    rc = z_center(slab=slab)

    # Check the coordination in the bulk first shell
    bulk_first_shell = []
    for atom in bulk:
        bulk_first_shell.append(
            [atom.type, len(neighbors(structure=bulk, nmax=1, center=atom.pos, tolerance=1e-1 / bulk.scale))])

    broken_bonds = []

    for i in range(len(under_coord)):
        atom = slab[under_coord[i][0]]
        coordination = under_coord[i][1]

        for j in range(len(bulk)):
            eq_site = float(atom.site - j) / float(len(bulk))
            if abs(int(eq_site) - eq_site) < 1e-5:
                break

        broken_bonds.append(abs(coordination - bulk_first_shell[j][1]))

    if len(broken_bonds) == 0:
        return 0.
    else:
        return sum(array(broken_bonds)) / float(len(broken_bonds))

##########################################################################


def count_broken_bonds_per_area(bulk=None, slab=None):
    """Counts broken bonds per atom"""

    from pylada.crystal import neighbors

    under_coord = sort_under_coord(bulk=bulk, slab=slab)
    rc = z_center(slab=slab)

    # Check the coordination in the bulk first shell
    bulk_first_shell = []
    for atom in bulk:
        bulk_first_shell.append(
            [atom.type, len(neighbors(structure=bulk, nmax=1, center=atom.pos, tolerance=1e-1 / bulk.scale))])

    broken_bonds = []

    for i in range(len(under_coord)):
        atom = slab[under_coord[i][0]]
        coordination = under_coord[i][1]

        for j in range(len(bulk)):
            eq_site = float(atom.site - j) / float(len(bulk))
            if abs(int(eq_site) - eq_site) < 1e-5:
                break

        broken_bonds.append(abs(coordination - bulk_first_shell[j][1]))

    c = cross(slab.cell[0], slab.cell[1])
    area = sqrt(dot(c, c)) * float(slab.scale)**2

    return sum(array(broken_bonds)) / area / 2  # there are two surfaces

##########################################################################


def count_tot_broken_bonds(bulk=None, slab=None):
    """Counts total number of broken bonds"""

    from pylada.crystal import neighbors

    under_coord = sort_under_coord(bulk=bulk, slab=slab)

    rc = z_center(slab=slab)

    # Check the coordination in the bulk first shell
    bulk_first_shell = []
    for atom in bulk:
        bulk_first_shell.append(
            [atom.type, len(neighbors(structure=bulk, nmax=1, center=atom.pos, tolerance=1e-1 / bulk.scale))])

    broken_bonds = []

    for i in range(len(under_coord)):
        atom = slab[under_coord[i][0]]
        coordination = under_coord[i][1]

        for j in range(len(bulk)):
            eq_site = float(atom.site - j) / float(len(bulk))
            if abs(int(eq_site) - eq_site) < 1e-5:
                break

        broken_bonds.append(abs(coordination - bulk_first_shell[j][1]))

    return sum(array(broken_bonds))

##########################################################################


def move_to_minimize_broken_bonds(bulk=None, slab=None, vacuum=None):
    from copy import deepcopy

    if count_broken_bonds(bulk=bulk, slab=slab) > 1.:

        pom = deepcopy(slab)
        rc = z_center(slab=pom)
        under_coord = sort_under_coord(bulk=bulk, slab=pom)

        top_indices = [i for i in range(len(under_coord)) if slab[under_coord[i][0]].pos[2] > rc]
        top_lowest_coord = min([under_coord[i][1] for i in top_indices])

        bottom_indices = [i for i in range(len(under_coord)) if slab[under_coord[i][0]].pos[2] < rc]
        bottom_lowest_coord = min([under_coord[i][1] for i in bottom_indices])

        if top_lowest_coord <= bottom_lowest_coord:
            for i in top_indices:
                if under_coord[i][1] == top_lowest_coord:
                    shift_to_bottom(structure=pom, atom=under_coord[i][0], vacuum=vacuum)

        elif top_lowest_coord > bottom_lowest_coord:
            for i in bottom_indices:
                if under_coord[i][1] == bottom_lowest_coord:
                    shift_to_top(structure=pom, atom=under_coord[i][0], vacuum=vacuum)
        return pom

    elif count_broken_bonds(bulk=bulk, slab=slab) == 1.:
        return slab

##########################################################################


def minimize_broken_bonds(bulk=None, slab=None, vacuum=None, charge=None, minimize_total=True):
    from copy import deepcopy

    no_broken = count_broken_bonds(bulk=bulk, slab=slab)

    if no_broken == 1.:
        print("Nothing to do your slab is already there :)")
        return slab

    elif no_broken == 0.:
        print("Nothing to do your slab is already there :)")
        return slab

    else:
        pom1 = deepcopy(slab)
        pom2 = deepcopy(slab)
        pom1 = move_to_minimize_broken_bonds(bulk=bulk, slab=pom1, vacuum=vacuum)

        # Minimizes the total number of broken bonds
        if minimize_total:
            while count_tot_broken_bonds(bulk=bulk, slab=pom1) < count_tot_broken_bonds(bulk=bulk, slab=pom2):
                pom2 = deepcopy(pom1)
                pom1 = move_to_minimize_broken_bonds(bulk=bulk, slab=pom1, vacuum=vacuum)

            if count_tot_broken_bonds(bulk=bulk, slab=pom1) == count_tot_broken_bonds(bulk=bulk, slab=pom2):
                cond1 = min([x[1] for x in sort_under_coord(bulk=bulk, slab=pom2)]) < min(
                    [x[1] for x in sort_under_coord(bulk=bulk, slab=pom1)])
                cond2 = abs(dipole_moment(slab=pom1, charge=charge)) < abs(
                    dipole_moment(slab=pom2, charge=charge))
                if cond1 or cond2:
                    pom2 = deepcopy(pom1)

        # Minimizes the average (per undercoordinated/surface atom) number of broken bonds
        else:
            while count_broken_bonds(bulk=bulk, slab=pom1) < count_broken_bonds(bulk=bulk, slab=pom2):
                pom2 = deepcopy(pom1)
                pom1 = move_to_minimize_broken_bonds(bulk=bulk, slab=pom1, vacuum=vacuum)

            if count_broken_bonds(bulk=bulk, slab=pom1) == count_broken_bonds(bulk=bulk, slab=pom2):
                cond1 = min([x[1] for x in sort_under_coord(bulk=bulk, slab=pom2)]) < min(
                    [x[1] for x in sort_under_coord(bulk=bulk, slab=pom1)])
                cond2 = abs(dipole_moment(slab=pom1, charge=charge)) < abs(
                    dipole_moment(slab=pom2, charge=charge))
                if cond1 or cond2:
                    pom2 = deepcopy(pom1)

        return pom2
##########################################################################


def is_polar(slab=None, charge=None, tol=1e-2):
    """Models the z-component of the dipole moment using the charges that are provided 
       and returns True if the moment is non zero, and False otherwise

       :param slab:   pylada structure
       :param charge: dictionary with atom.types and keys and charges as values, 
                      only different atom types are needed
    """

    moment = dipole_moment(slab=slab, charge=charge)
    print("dipole_moment=", moment)

    if abs(moment) < tol:
        return False  # NON POLAR
    else:
        return True  # POLAR!!!
##########################################################################
##########################################################################
