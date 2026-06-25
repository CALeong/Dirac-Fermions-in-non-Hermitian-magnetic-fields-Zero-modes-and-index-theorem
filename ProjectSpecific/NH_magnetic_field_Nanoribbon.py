import numpy as np
import scipy
from ProjectSpecific.Honeycomb_Nanoribbon import honeycomb_nanoribbon_PBC


def identify_r_distance_honeycomb_nanoribbon(num_edge_sites, num_levels):
    if (num_levels % 2) == 0:
        raise ValueError

    r_vals_for_each_gen = np.arange(num_levels // 2, -0.1, -1)
    r_vals_for_each_gen = np.append(r_vals_for_each_gen, np.flip(r_vals_for_each_gen[:-1]))

    r_vals = np.array([])
    for r in r_vals_for_each_gen:
        r_vals = np.append(r_vals, np.repeat(r, num_edge_sites))

    return r_vals


def type_1_nonhermitian_magnetic_field_nanoribbon(num_edge_sites, num_levels, beta=1.0, field_profile='uniform'):
    # Nearest-neighbor hamiltonian in sublattice basis
    tbham = honeycomb_nanoribbon_PBC(num_edge_sites, num_levels)

    # r values, whose ordering is converted to sublattice basis
    r_vals = identify_r_distance_honeycomb_nanoribbon(num_edge_sites, num_levels)

    if field_profile == 'uniform':
        r_vals = r_vals ** 2
    else:
        raise ValueError

    # form final hamiltonian
    return np.diag(np.exp(beta * r_vals)) @ tbham @ np.diag(np.exp(-beta * r_vals))


def get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC(num_edge_sites, num_levels):

    if (num_edge_sites % 2) == 1:
        raise ValueError

    all_plaquette_boundary_sites = np.zeros((1, 6), dtype=np.int64)
    new_plaquette_boundary_sites = np.zeros((num_edge_sites // 2, 1), dtype=np.int64)
    for nl in range(num_levels):
        sites_on_level = np.arange(nl * num_edge_sites, (nl + 1) * num_edge_sites)
        split_sites = np.array(np.split(sites_on_level, num_edge_sites // 2))
        new_plaquette_boundary_sites = np.hstack((new_plaquette_boundary_sites, split_sites))
        if new_plaquette_boundary_sites.shape[1] == 7:
            all_plaquette_boundary_sites = np.vstack((all_plaquette_boundary_sites, new_plaquette_boundary_sites[:, 1:]))
            new_plaquette_boundary_sites = np.zeros((num_edge_sites // 2, 1), dtype=np.int64)
            new_plaquette_boundary_sites = np.hstack((new_plaquette_boundary_sites, split_sites))

    new_plaquette_boundary_sites = np.zeros((num_edge_sites // 2 - 1, 1), dtype=np.int64)
    for nl in range(1, num_levels):
        relevant_sites_on_level = np.arange(num_edge_sites * nl + 1, (nl + 1) * num_edge_sites - 1)
        split_sites = np.array(np.split(relevant_sites_on_level, num_edge_sites // 2 - 1))
        new_plaquette_boundary_sites = np.hstack((new_plaquette_boundary_sites, split_sites))
        if new_plaquette_boundary_sites.shape[1] == 7:
            all_plaquette_boundary_sites = np.vstack((all_plaquette_boundary_sites, new_plaquette_boundary_sites[:, 1:]))
            new_plaquette_boundary_sites = np.zeros((num_edge_sites // 2 - 1, 1), dtype=np.int64)
            new_plaquette_boundary_sites = np.hstack((new_plaquette_boundary_sites, split_sites))

    new_plaquette_boundary_sites = np.array([], dtype=np.int64)
    for nl in range(1, num_levels):
        relevant_sites_on_level = np.array([(nl + 1) * num_edge_sites - 1, num_edge_sites * nl])
        new_plaquette_boundary_sites = np.append(new_plaquette_boundary_sites, relevant_sites_on_level)
        if len(new_plaquette_boundary_sites) == 6:
            all_plaquette_boundary_sites = np.vstack((all_plaquette_boundary_sites, new_plaquette_boundary_sites))
            new_plaquette_boundary_sites = np.array([], dtype=np.int64)
            new_plaquette_boundary_sites = np.append(new_plaquette_boundary_sites, relevant_sites_on_level)

    all_plaquette_boundary_sites = all_plaquette_boundary_sites[1:, :]
    ccw_site_ordering = [0, 1, 3, 5, 4, 2]
    all_plaquette_boundary_sites = all_plaquette_boundary_sites[:, ccw_site_ordering]

    return all_plaquette_boundary_sites


def honeycomb_nanoribbon_site_assignment(num_edge_sites, num_levels):
    asites = np.array([], dtype=np.int64)
    bsites = np.array([], dtype=np.int64)
    for nl in range(num_levels):
        if (nl % 2) == 0:
            asites = np.append(asites, np.arange(nl * num_edge_sites, (nl + 1) * num_edge_sites, 2))
            bsites = np.append(bsites, np.arange(nl * num_edge_sites + 1, (nl + 1) * num_edge_sites, 2))
        else:
            asites = np.append(asites, np.arange(nl * num_edge_sites + 1, (nl + 1) * num_edge_sites, 2))
            bsites = np.append(bsites, np.arange(nl * num_edge_sites, (nl + 1) * num_edge_sites, 2))

    return asites, bsites


def haldane_current_honeycomb_nanoribbon_PBC(num_edge_sites, num_levels):
    plaquette_boundary_sites = get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC(num_edge_sites, num_levels)
    asites, bsites = honeycomb_nanoribbon_site_assignment(num_edge_sites, num_levels)
    asites_mask = np.isin(plaquette_boundary_sites, asites)
    plaquette_boundary_sites_signs = plaquette_boundary_sites.copy()
    plaquette_boundary_sites_signs[asites_mask] = 1
    plaquette_boundary_sites_signs[~asites_mask] = -1

    total_num_sites = num_edge_sites * num_levels
    hamiltonian = np.zeros((total_num_sites, total_num_sites), dtype=np.complex128)

    # plaquette_boundary_sites[:, [0, 2]]
    isites = plaquette_boundary_sites[:, 0]
    jsites = plaquette_boundary_sites[:, 2]
    hamiltonian[isites, jsites] = plaquette_boundary_sites_signs[:, 0] * 1j

    # plaquette_boundary_sites[:, [1, 3]]
    isites = plaquette_boundary_sites[:, 1]
    jsites = plaquette_boundary_sites[:, 3]
    hamiltonian[isites, jsites] = plaquette_boundary_sites_signs[:, 1] * 1j

    # plaquette_boundary_sites[:, [2, 4]]
    isites = plaquette_boundary_sites[:, 2]
    jsites = plaquette_boundary_sites[:, 4]
    hamiltonian[isites, jsites] = plaquette_boundary_sites_signs[:, 2] * 1j

    # plaquette_boundary_sites[:, [3, 5]]
    isites = plaquette_boundary_sites[:, 3]
    jsites = plaquette_boundary_sites[:, 5]
    hamiltonian[isites, jsites] = plaquette_boundary_sites_signs[:, 3] * 1j

    # plaquette_boundary_sites[:, [4, 0]]
    isites = plaquette_boundary_sites[:, 4]
    jsites = plaquette_boundary_sites[:, 0]
    hamiltonian[isites, jsites] = plaquette_boundary_sites_signs[:, 4] * 1j

    # plaquette_boundary_sites[:, [5, 1]]
    isites = plaquette_boundary_sites[:, 5]
    jsites = plaquette_boundary_sites[:, 1]
    hamiltonian[isites, jsites] = plaquette_boundary_sites_signs[:, 5] * 1j

    hamiltonian = hamiltonian + hamiltonian.T.conj()

    return hamiltonian


def type_2_nonhermitian_magnetic_field_generation_based_nanoribbon(num_edge_sites, num_levels, beta=1.0,
                                                                   field_profile='uniform',
                                                                   return_chi_term=False):

    # Nearest-neighbor hamiltonian in counting basis
    tbham = honeycomb_nanoribbon_PBC(num_edge_sites, num_levels)

    # Form Haldane term and convert to sublattice basis
    haldane_ham = haldane_current_honeycomb_nanoribbon_PBC(num_edge_sites, num_levels)
    asites, bsites = honeycomb_nanoribbon_site_assignment(num_edge_sites, num_levels)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    haldane_ham = haldane_ham[:, sublattice_basis][sublattice_basis, :]

    # Reverse B-B current direction
    halfway = int(haldane_ham.shape[0] / 2)
    haldane_ham[halfway:, halfway:] = np.conj(haldane_ham[halfway:, halfway:])

    # back to counting basis
    haldane_ham = haldane_ham[:, np.argsort(sublattice_basis)][np.argsort(sublattice_basis), :]

    r_vals = identify_r_distance_honeycomb_nanoribbon(num_edge_sites, num_levels)
    r_vals = r_vals ** 2

    # Get plaquette boundary sites
    plaq_bound_sites = get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC(num_edge_sites, num_levels)

    # Form the matrices to be exponentiated
    # haldane_ham used to ensure proper signs and make imaginary
    # r values associated with each bond minimum of the two associated r values
    exp_plus = np.zeros(haldane_ham.shape, dtype=np.complex128)
    exp_minus = np.zeros(haldane_ham.shape, dtype=np.complex128)
    for row in range(plaq_bound_sites.shape[0]):
        single_plaquette_bound_sites = plaq_bound_sites[row, :]
        min_r_val = np.min(r_vals[single_plaquette_bound_sites])
        isites = single_plaquette_bound_sites
        jsites = np.roll(single_plaquette_bound_sites, -2)
        exp_plus[isites, jsites] = beta * haldane_ham[isites, jsites] * min_r_val
        exp_plus[jsites, isites] = beta * haldane_ham[jsites, isites] * min_r_val
        exp_minus[isites, jsites] = -beta * haldane_ham[isites, jsites] * min_r_val
        exp_minus[jsites, isites] = -beta * haldane_ham[jsites, isites] * min_r_val

    if return_chi_term:
        return exp_plus
    else:
        return scipy.linalg.expm(exp_plus) @ tbham @ scipy.linalg.expm(exp_minus)  # noqa


