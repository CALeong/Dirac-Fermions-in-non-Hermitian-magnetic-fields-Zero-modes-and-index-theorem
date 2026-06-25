import numpy as np
from Lattice.Honeycomb_Sparse import honeycomb_points


# From Axial_Magnetic_Field.Honeycomb

def points_on_level(level):
    sites_per_level = honeycomb_points(level)[0]
    #     print(sites_per_level)
    #     print(np.arange(np.sum(sites_per_level[:level-1]), np.sum(sites_per_level[:level])))
    return (np.arange(np.sum(sites_per_level[:level - 1]), np.sum(sites_per_level[:level])))


def corner_points_on_level(level):
    num_points_on_side = 2 * level - 1
    first_site_on_gen = np.sum(honeycomb_points(level)[0][:-1])
    first_corner_point = first_site_on_gen + (num_points_on_side - 1) / 2
    corner_points = np.array([first_corner_point, first_corner_point + 1])
    for i in range(5):
        corner_points = np.append(corner_points, corner_points[-2:] + num_points_on_side)
    return (corner_points)


def connected_to_next_gen_points(level):
    connected_to_next_gen_points_list = np.array([], dtype=int)
    corner_points = corner_points_on_level(level)
    first_site_on_gen = np.sum(honeycomb_points(level)[0][:-1])

    if level % 2 == 0:
        start_point = first_site_on_gen + 1
        for i in range(0, len(corner_points), 2):
            connected_to_next_gen_points_list = np.append(connected_to_next_gen_points_list,
                                                          np.arange(start_point, corner_points[i] + 1, 2, dtype=int))
            start_point = corner_points[i + 1]

        connected_to_next_gen_points_list = np.append(connected_to_next_gen_points_list,
                                                      np.arange(start_point, honeycomb_points(level)[1], 2, dtype=int))

    elif level % 2 == 1:
        start_point = first_site_on_gen
        for i in range(0, len(corner_points), 2):
            connected_to_next_gen_points_list = np.append(connected_to_next_gen_points_list,
                                                          np.arange(start_point, corner_points[i] + 1, 2, dtype=int))
            start_point = corner_points[i + 1]

        connected_to_next_gen_points_list = np.append(connected_to_next_gen_points_list,
                                                      np.arange(start_point, honeycomb_points(level)[1], 2, dtype=int))
    return (connected_to_next_gen_points_list)


def connected_to_prev_gen_points(level):
    points_on_layer = np.arange(np.sum((honeycomb_points(level)[0])[:-1]), honeycomb_points(level)[1])
    points_connected_to_next_gen = connected_to_next_gen_points(level)
    return (np.array([int(i) for i in np.setdiff1d(points_on_layer, points_connected_to_next_gen)]))


# From Fundamental.Local

def get_nnn_hoppings_around_plaquet(plaquet_boundary_sites):
    nnn_hoppings_by_plaquet_isites = np.zeros(np.shape(plaquet_boundary_sites), dtype=int)
    nnn_hoppings_by_plaquet_jsites = np.zeros(np.shape(plaquet_boundary_sites), dtype=int)
    for row in range(np.size(plaquet_boundary_sites,0)):
        sites = plaquet_boundary_sites[row, :]
        plaquet_nnn_hops_isites = np.array([])
        plaquet_nnn_hops_jsites = np.array([])
        for s in range(len(sites)):
            plaquet_nnn_hops_isites = np.append(plaquet_nnn_hops_isites, np.take(sites, s, mode='wrap'))
            plaquet_nnn_hops_jsites = np.append(plaquet_nnn_hops_jsites, np.take(sites, s+2, mode='wrap'))
        nnn_hoppings_by_plaquet_isites[row, :] = plaquet_nnn_hops_isites
        nnn_hoppings_by_plaquet_jsites[row, :] = plaquet_nnn_hops_jsites
    return(nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites)


def get_plaquet_boundary_sites_honeycomb(num_levels):
    plaq_bound_sites = np.arange(0, 6, dtype=int) #hard code first generation
    for nl in range(1, num_levels):
        points_conn_next_gen = connected_to_next_gen_points(nl)
        points_conn_prev_gen = connected_to_prev_gen_points(nl + 1)
        for i in range(0, len(points_conn_next_gen)-1):
            bottom = np.flip(np.arange(points_conn_next_gen[i], points_conn_next_gen[i+1]+1, dtype=int))
            top = np.arange(points_conn_prev_gen[i], points_conn_prev_gen[i+1]+1, dtype=int)
            plaq_bound_sites = np.vstack((plaq_bound_sites, np.concatenate((bottom, top))))
        #Handle last plaquet on each level
        sites_on_level = points_on_level(nl).astype(int)
        sites_on_next_level = points_on_level(nl + 1).astype(int)
        bottom = np.flip(np.concatenate((sites_on_level[np.where(points_conn_next_gen[-1] == sites_on_level)[0][0]:],
                                 sites_on_level[:np.where(points_conn_next_gen[0] == sites_on_level)[0][0]+1])))
        top = np.concatenate((sites_on_next_level[np.where(points_conn_prev_gen[-1] == sites_on_next_level)[0][0]:],
                                 sites_on_next_level[:np.where(points_conn_prev_gen[0] == sites_on_next_level)[0][0]+1]))
        bottom = np.array([int(b) for b in bottom])
        top = np.array([int(t) for t in top])
        plaq_bound_sites = np.vstack((plaq_bound_sites, np.concatenate((bottom, top))))
    return(plaq_bound_sites)