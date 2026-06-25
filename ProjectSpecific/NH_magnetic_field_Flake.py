import numpy as np
import scipy
from Lattice.Honeycomb_Sparse import (honeycomb_points,
                                      sites_conn_to_next_gen,
                                      sites_conn_to_prev_gen,
                                      honeycomb_first_site_on_gen,
                                      honeycomb_site_assignment,
                                      honeycomb_lattice_sparse)
from Lattice.Hamiltonians_Sublattice_Basis import hamiltonian_honeycombOBC_sublattice_basis
from ProjectSpecific.Other_Honeycomb import connected_to_next_gen_points, connected_to_prev_gen_points


def identify_r_distance_honeycomb(nval):
    total_num_sites = honeycomb_points(nval)[1]

    first_point_on_level = honeycomb_first_site_on_gen(nval)

    r_vals = np.repeat(-1, total_num_sites)  # Populate r_vals with sentinel value -1
    r_vals[:6] = 0  # hard code first generation as r=0
    for n in range(1, nval):
        # Get intergen sites between nth generation and n-1th generation (anchor points)
        curr_gen_conn_below = sites_conn_to_prev_gen(n, first_point_on_level)
        prev_gen_conn_above = sites_conn_to_next_gen(n - 1, first_point_on_level)

        # if/else to check assumption that all anchor points on each generation
        # have their own single unique respective r value
        if len(np.unique(r_vals[prev_gen_conn_above])) == 1:
            # Get unique r value of n-1th generation anchor points to assign r values for nth generation r values
            unique_prev_gen_conn_above_r_val = np.unique(r_vals[prev_gen_conn_above])[0]
            r_vals[curr_gen_conn_below] = unique_prev_gen_conn_above_r_val + 1

            # On each generation all sites are inter-generation connected but some only connect to
            # previous generation and some only connect to above generation
            # Previous block handled teh former, current block handles the latter
            sites_on_level = np.arange(first_point_on_level[n], first_point_on_level[n + 1], dtype=np.int64)
            remaining_curr_gen_sites = np.setdiff1d(sites_on_level, curr_gen_conn_below)
            r_vals[remaining_curr_gen_sites] = unique_prev_gen_conn_above_r_val + 2
        else:
            raise ValueError

    return r_vals


def type_1_nonhermitian_magnetic_field(nval, beta=1.0, field_profile='uniform'):
    # Nearest-neighbor hamiltonian in sublattice basis
    tbham = hamiltonian_honeycombOBC_sublattice_basis(nval).toarray()

    # r values, whose ordering is converted to sublattice basis
    r_vals = identify_r_distance_honeycomb(nval)
    asites, bsites = honeycomb_site_assignment(nval)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    r_vals = r_vals[sublattice_basis]
    if field_profile == 'uniform':
        r_vals = r_vals ** 2
    else:
        raise ValueError

    # form final hamiltonian
    return np.diag(np.exp(beta * r_vals)) @ tbham @ np.diag(np.exp(-beta * r_vals))


def haldane_current_honeycomb(num_levels, tight_bind_ham, t2):
    haldane_mat = np.copy(tight_bind_ham).astype(np.complex128)

    points_per_level = honeycomb_points(num_levels)[0]

    # points_conn_next_level = sites on current gen that connect to above gen sites
    # points_that_conn_from_above = sites on next gen that connect down to current gen

    # Hard code first gen since get_if_point_is_connected_with_upper_layer_q3_general does not work for first gen
    sites_on_level = np.arange(0, 6, dtype=int)
    sites_on_next_level = np.arange(6, np.sum(points_per_level[:2]), dtype=int)
    points_conn_next_level = connected_to_next_gen_points(1)
    points_that_conn_from_above = connected_to_prev_gen_points(2)
    for i in range(6):  # intragen nnn hopping
        haldane_mat[i, np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
        haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), i] = np.conj(1j * t2)
    for i in range(len(points_conn_next_level)):  # intergen nnn hopping
        next_level_site_connected_index = np.where(sites_on_next_level == points_that_conn_from_above[i])[0]
        nnn_sites_plusone = np.take(sites_on_next_level, next_level_site_connected_index + 1, mode='wrap')
        nnn_sites_minusone = np.take(sites_on_next_level, next_level_site_connected_index - 1, mode='wrap')
        haldane_mat[points_conn_next_level[i], nnn_sites_plusone] = 1j * t2
        haldane_mat[nnn_sites_plusone, points_conn_next_level[i]] = np.conj(1j * t2)
        haldane_mat[points_conn_next_level[i], nnn_sites_minusone] = np.conj(1j * t2)
        haldane_mat[nnn_sites_minusone, points_conn_next_level[i]] = 1j * t2
        # Finally need to take into account nnn hopping for adjacent points_that_conn_from_above
        nnn_sites_plusplusone = np.take(points_that_conn_from_above, i + 1, mode='wrap')
        nnn_sites_minusminusone = np.take(points_that_conn_from_above, i - 1, mode='wrap')
        haldane_mat[points_conn_next_level[i], nnn_sites_plusplusone] = np.conj(1j * t2)
        haldane_mat[nnn_sites_plusplusone, points_conn_next_level[i]] = 1j * t2
        haldane_mat[points_conn_next_level[i], nnn_sites_minusminusone] = 1j * t2
        haldane_mat[nnn_sites_minusminusone, points_conn_next_level[i]] = np.conj(1j * t2)

    for nl in range(1, num_levels - 1):  # Handles all other generations except the last gen
        sites_on_level = np.arange(np.sum(points_per_level[:nl]), np.sum(points_per_level[:nl + 1]), dtype=int)
        sites_on_next_level = np.arange(np.sum(points_per_level[:nl + 1]), np.sum(points_per_level[:nl + 2]), dtype=int)
        points_conn_next_level = connected_to_next_gen_points(nl + 1)
        points_that_conn_from_above = connected_to_prev_gen_points(nl + 2)
        for i in range(len(sites_on_level)):  # intragen nnn hopping
            if np.take(sites_on_level, i + 1, mode='wrap') in points_conn_next_level: #Need this due to honeycomb geometry
                haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
                haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = np.conj(1j * t2)
            else:
                haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = np.conj(1j * t2)
                haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = 1j * t2

        # intergen nnn hopping
        for i in range(len(points_conn_next_level)):
            next_level_site_connected_index = np.where(sites_on_next_level == points_that_conn_from_above[i])[0]
            nnn_sites_plusone = np.take(sites_on_next_level, next_level_site_connected_index + 1, mode='wrap')
            nnn_sites_minusone = np.take(sites_on_next_level, next_level_site_connected_index - 1, mode='wrap')
            haldane_mat[points_conn_next_level[i], nnn_sites_plusone] = 1j * t2
            haldane_mat[nnn_sites_plusone, points_conn_next_level[i]] = np.conj(1j * t2)
            haldane_mat[points_conn_next_level[i], nnn_sites_minusone] = np.conj(1j * t2)
            haldane_mat[nnn_sites_minusone, points_conn_next_level[i]] = 1j * t2
            # Modified code to account nnn hopping for adjacent points_that_conn_from_above for honeycomb geometry
            if np.take(points_conn_next_level, i+1, mode='wrap') - points_conn_next_level[i] == 1:
                nnn_sites_plusplusone = np.take(points_that_conn_from_above, i + 1, mode='wrap')
                haldane_mat[points_conn_next_level[i], nnn_sites_plusplusone] = np.conj(1j * t2)
                haldane_mat[nnn_sites_plusplusone, points_conn_next_level[i]] = 1j * t2
            if points_conn_next_level[i] - points_conn_next_level[i-1] == 1:
                nnn_sites_minusminusone = np.take(points_that_conn_from_above, i - 1, mode='wrap')
                haldane_mat[points_conn_next_level[i], nnn_sites_minusminusone] = 1j * t2
                haldane_mat[nnn_sites_minusminusone, points_conn_next_level[i]] = np.conj(1j * t2)
        # Finally need to account for sites on gen that do not connect via NN hopping to next gen
        points_not_conn_next_level = np.setdiff1d(sites_on_level, points_conn_next_level)
        for pncnl in points_not_conn_next_level:
            pncnl_index = np.where(sites_on_level == pncnl)[0]
            siteoneabove = np.take(sites_on_level, pncnl_index + 1, mode='wrap')
            siteonebelow = np.take(sites_on_level, pncnl_index - 1, mode='wrap')
            nnnnextgenoneabovesite = points_that_conn_from_above[np.where(points_conn_next_level == siteoneabove)[0]]
            nnnnextgenonebelowsite = points_that_conn_from_above[np.where(points_conn_next_level == siteonebelow)[0]]
            haldane_mat[pncnl, nnnnextgenoneabovesite] = np.conj(1j * t2)
            haldane_mat[nnnnextgenoneabovesite, pncnl] = 1j * t2
            haldane_mat[pncnl, nnnnextgenonebelowsite] = 1j * t2
            haldane_mat[nnnnextgenonebelowsite, pncnl] = np.conj(1j * t2)

    # Finally handle last gen
    points_conn_next_level = connected_to_next_gen_points(num_levels)
    sites_on_level = np.arange(np.sum(points_per_level[:num_levels - 1]), np.sum(points_per_level), dtype=int)
    for i in range(len(sites_on_level)):  # intragen nnn hopping (all intergen hopping already taken care of)
        # if (sites_on_level[i] in points_conn_next_level) and (np.take(sites_on_level, i+1, mode='wrap') not in points_conn_next_level):
        #     haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = np.conj(1j * t2)
        #     haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = 1j * t2
        # else:
        #     haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
        #     haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = np.conj(1j * t2)
        haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
        haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = np.conj(1j * t2)

    #Last bit of code to reverse sign of phase for NNN hopping on B sublattice
    asites, bsites = honeycomb_site_assignment(num_levels)
    bsites = [int(b) for b in bsites]
    haldane_mat[bsites, :] = np.conj(haldane_mat[bsites, :])

    return (haldane_mat)

def haldane_current_termonly_honeycomb(num_levels, tight_bind_ham, t2):
    wholeham = haldane_current_honeycomb(num_levels, tight_bind_ham, t2)
    return(wholeham - tight_bind_ham)


def type_2_nonhermitian_magnetic_field_r_average(nval, beta=1.0, field_profile='uniform'):

    # Nearest-neighbor hamiltonian in sublattice basis
    tbham = hamiltonian_honeycombOBC_sublattice_basis(nval).toarray()

    # Form Haldane term and convert to sublattice basis
    haldane_ham = haldane_current_honeycomb(nval, tbham, 1)
    asites, bsites = honeycomb_site_assignment(nval)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    haldane_ham = haldane_ham[:, sublattice_basis][sublattice_basis, :]

    # Reverse B-B current direction
    halfway = int(haldane_ham.shape[0] / 2)
    haldane_ham[halfway:, halfway:] = np.conj(haldane_ham[halfway:, halfway:])

    # r values, whose ordering is converted to sublattice basis
    r_vals = identify_r_distance_honeycomb(nval)
    r_vals = r_vals[sublattice_basis]
    if field_profile == 'uniform':
        r_vals = r_vals ** 2
    else:
        raise ValueError

    # identify next-nearest-neighbor hopping pairs
    nnn_i, nnn_j = haldane_ham.nonzero()

    # Form the matrices to be exponentiated
    # haldane_ham used to ensure proper signs and make imaginary
    # r values associated with each bond is averaged to give magnitude
    exp_plus = haldane_ham.copy()
    exp_plus[nnn_i, nnn_j] = beta * haldane_ham[nnn_i, nnn_j] * (r_vals[nnn_i] + r_vals[nnn_j]) / 2
    exp_minus = haldane_ham.copy()
    exp_minus[nnn_i, nnn_j] = -beta * haldane_ham[nnn_i, nnn_j] * (r_vals[nnn_i] + r_vals[nnn_j]) / 2

    return scipy.linalg.expm(exp_plus) @ tbham @ scipy.linalg.expm(exp_minus)  # noqa


def type_2_nonhermitian_magnetic_field_generation_based(nval, beta=1.0, field_profile='uniform'):

    # Nearest-neighbor hamiltonian in sublattice basis
    tbham = hamiltonian_honeycombOBC_sublattice_basis(nval).toarray()

    # Form Haldane term and convert to sublattice basis
    haldane_ham = haldane_current_honeycomb(nval, tbham, 1)
    asites, bsites = honeycomb_site_assignment(nval)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    haldane_ham = haldane_ham[:, sublattice_basis][sublattice_basis, :]

    # Reverse B-B current direction
    halfway = int(haldane_ham.shape[0] / 2)
    haldane_ham[halfway:, halfway:] = np.conj(haldane_ham[halfway:, halfway:])

    # r values, whose ordering is converted to sublattice basis
    # Unlike elsewhere, not bond lengths from center, instead associated with generation site belongs to
    num_sites_each_level = honeycomb_points(nval)[0]
    r_vals = np.array([])
    for i, n in enumerate(num_sites_each_level):
        r_vals = np.append(r_vals, np.repeat(i, n))
    r_vals = r_vals[sublattice_basis]

    # identify next-nearest-neighbor hopping pairs
    nnn_i, nnn_j = haldane_ham.nonzero()

    # Form the matrices to be exponentiated
    # haldane_ham used to ensure proper signs and make imaginary
    # r values associated with each bond minimum of the two associated r values
    exp_plus = haldane_ham.copy()
    exp_plus[nnn_i, nnn_j] = beta * haldane_ham[nnn_i, nnn_j] * np.minimum(r_vals[nnn_i], r_vals[nnn_j])
    exp_minus = haldane_ham.copy()
    exp_minus[nnn_i, nnn_j] = -beta * haldane_ham[nnn_i, nnn_j] * np.minimum(r_vals[nnn_i], r_vals[nnn_j])

    return scipy.linalg.expm(exp_plus) @ tbham @ scipy.linalg.expm(exp_minus)  # noqa

