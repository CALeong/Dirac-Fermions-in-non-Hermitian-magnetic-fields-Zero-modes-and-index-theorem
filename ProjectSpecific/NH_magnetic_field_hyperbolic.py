import numpy as np
import scipy
from Lattice.General_Hamiltonian import general_hyperbolic_q3_hamiltonian, number_points_q3_general_from_repeating_pattern
from Lattice.General_Hamiltonian import get_sites_conn_to_below_gen, get_sites_conn_to_above_gen, first_site_on_gen_hyperbolic_q3


def type1_nonhermitian_uniform_magnetic_field_hyperbolic(pval, nval, beta):
    points_per_level = number_points_q3_general_from_repeating_pattern(pval, nval)[0]
    r_vals = np.array([])
    for n in range(nval):
        r_vals = np.append(r_vals, np.repeat(n ** 2, points_per_level[n]))
    exp_plus = np.diag(np.exp(beta * r_vals))
    exp_minus = np.diag(np.exp(-beta * r_vals))
    tbham = general_hyperbolic_q3_hamiltonian(pval, nval).toarray()
    return exp_plus @ tbham @ exp_minus


def hyperbolic_plaquette_boundary_sites_q3(pval, nval):
    plaq_bound_sites = np.flip(np.arange(0, pval)).reshape(1, -1)
    first_sites_on_gen = first_site_on_gen_hyperbolic_q3(pval, nval)
    for n in range(1, nval):
        curr_gen_conn_above = get_sites_conn_to_above_gen(pval, n, first_sites_on_gen)
        next_gen_conn_below = get_sites_conn_to_below_gen(pval, n + 1, first_sites_on_gen)
        for i in range(len(curr_gen_conn_above) - 1):
            curr_gen_bound_sites = np.arange(curr_gen_conn_above[i], curr_gen_conn_above[i + 1] + 1).astype(np.int64)
            next_gen_bound_sites = np.arange(next_gen_conn_below[i], next_gen_conn_below[i + 1] + 1).astype(np.int64)
            plaq_bound_sites = np.vstack((plaq_bound_sites,
                                          np.concat((curr_gen_bound_sites, np.flip(next_gen_bound_sites)))))
        curr_gen_bound_sites = np.arange(curr_gen_conn_above[-1], first_sites_on_gen[n])
        curr_gen_bound_sites = np.append(curr_gen_bound_sites,
                                         np.arange(first_sites_on_gen[n - 1], curr_gen_conn_above[0] + 1))
        next_gen_bound_sites = np.arange(next_gen_conn_below[-1], first_sites_on_gen[n + 1])
        next_gen_bound_sites = np.append(next_gen_bound_sites,
                                         np.arange(first_sites_on_gen[n], next_gen_conn_below[0] + 1))
        plaq_bound_sites = np.vstack((plaq_bound_sites,
                                      np.concat((curr_gen_bound_sites, np.flip(next_gen_bound_sites)))))
    return plaq_bound_sites.astype(np.int64)


def type2_nonhermitian_uniform_magnetic_field_hyperbolic_q3(pval, nval, beta):
    points_per_level, total_num_points = number_points_q3_general_from_repeating_pattern(pval, nval)
    r_vals = np.array([])
    for n in range(nval):
        r_vals = np.append(r_vals, np.repeat(n ** 2, points_per_level[n]))

    chi_ham = np.zeros((total_num_points, total_num_points), np.complex128)

    plaq_bound_sites = hyperbolic_plaquette_boundary_sites_q3(pval, nval)
    for row in range(plaq_bound_sites.shape[0]):
        isites = plaq_bound_sites[row, :]
        jsites = np.roll(plaq_bound_sites[row, :], -2)
        plaquette_min_rval = np.min(r_vals[plaq_bound_sites[row, :]])
        chi_ham[isites, jsites] = plaquette_min_rval
        chi_ham[jsites, isites] = -1 * plaquette_min_rval

    exp_plus = scipy.linalg.expm(beta * 1j * chi_ham)
    exp_minus = scipy.linalg.expm(beta * -1j * chi_ham)

    tbham = general_hyperbolic_q3_hamiltonian(pval, nval).toarray()
    return exp_plus @ tbham @ exp_minus  # noqa



