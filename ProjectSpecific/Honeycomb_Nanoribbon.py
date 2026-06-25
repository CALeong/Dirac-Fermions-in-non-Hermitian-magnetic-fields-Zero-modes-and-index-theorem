import numpy as np


def honeycomb_nanoribbon_PBC(num_edge_sites, num_levels):
    if (num_edge_sites % 2) != 0:
        raise ValueError

    total_num_sites = int(num_edge_sites * num_levels)
    hamiltonian = np.zeros((total_num_sites, total_num_sites))
    first_site_each_level = np.cumsum(np.repeat(num_edge_sites, num_levels)).astype(np.int64)
    first_site_each_level = np.append(np.array([0]), first_site_each_level)
    for nl in range(num_levels - 1):
        # intralayer hoppings
        i_sites = np.arange(first_site_each_level[nl], first_site_each_level[nl + 1], 2).astype(np.int64)
        j_sites = np.arange(first_site_each_level[nl] + 1, first_site_each_level[nl + 1], 2).astype(np.int64)
        if (nl % 2) == 0:
            hamiltonian[i_sites, j_sites] = 1
            hamiltonian[j_sites, i_sites] = 1
        else:  # PBC hoppings naturally included here
            hamiltonian[np.roll(i_sites, shift=-1), j_sites] = 1
            hamiltonian[j_sites, np.roll(i_sites, shift=-1)] = 1

        # interlayer hoppings
        sites_on_level = np.arange(first_site_each_level[nl], first_site_each_level[nl + 1]).astype(np.int64)
        sites_on_next_level = np.arange(first_site_each_level[nl + 1], first_site_each_level[nl + 2]).astype(np.int64)
        hamiltonian[sites_on_level, sites_on_next_level] = 1
        hamiltonian[sites_on_next_level, sites_on_level] = 1

    # intralayer hoppings for last layer
    i_sites = np.arange(first_site_each_level[num_levels - 1], first_site_each_level[num_levels], 2).astype(np.int64)
    j_sites = np.arange(first_site_each_level[num_levels - 1] + 1, first_site_each_level[num_levels], 2).astype(np.int64)
    hamiltonian[i_sites, j_sites] = 1
    hamiltonian[j_sites, i_sites] = 1

    return hamiltonian


def plot_honeycomb_nanoribbon(num_edge_sites, num_levels):
    if (num_edge_sites % 2) != 0:
        raise ValueError

    b1vec = np.array([1 / (2 * np.sqrt(3)), 1 / 2])
    b2vec = np.array([1 / (2 * np.sqrt(3)), -1 / 2])
    b3vec = np.array([-1 / np.sqrt(3), 0])

    sites = np.zeros((1, 2), dtype=np.float64)
    sites = np.vstack((sites, sites[0, :] - b3vec))
    for nl in range(num_levels):

        for n in range(int(num_edge_sites / 2) - 1):
            sites = np.vstack((sites, sites[-2:, :] + b1vec - 2 * b3vec + b2vec))
        if nl != (num_levels - 1):
            if (nl % 2) == 0:
                sites = np.vstack((sites, sites[-num_edge_sites, :] - b2vec))
                sites = np.vstack((sites, sites[-num_edge_sites, :] + b1vec))
            else:
                sites = np.vstack((sites, sites[-num_edge_sites, :] + b1vec))
                sites = np.vstack((sites, sites[-num_edge_sites, :] - b2vec))

    tbham = honeycomb_nanoribbon_PBC(num_edge_sites, num_levels)
    total_num_sites = int(num_edge_sites * num_levels)
    left_pbc_sites = np.arange(num_edge_sites, total_num_sites, 2 * num_edge_sites).astype(np.int64)
    right_pbc_sites = np.arange(2 * num_edge_sites - 1, total_num_sites, 2 * num_edge_sites).astype(np.int64)
    tbham[left_pbc_sites, right_pbc_sites] = 0
    tbham[right_pbc_sites, left_pbc_sites] = 0

    conns = tbham.nonzero()

    return sites, conns
