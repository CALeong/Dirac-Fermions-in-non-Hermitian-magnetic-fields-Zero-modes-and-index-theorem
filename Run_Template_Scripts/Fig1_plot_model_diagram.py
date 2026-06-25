# Import dependencies
import numpy as np
from ProjectSpecific.Honeycomb_Nanoribbon import plot_honeycomb_nanoribbon
from ProjectSpecific.NH_magnetic_field_Nanoribbon import haldane_current_honeycomb_nanoribbon_PBC
from ProjectSpecific.NH_magnetic_field_Nanoribbon import identify_r_distance_honeycomb_nanoribbon
from ProjectSpecific.NH_magnetic_field_Nanoribbon import honeycomb_nanoribbon_site_assignment
from ProjectSpecific.NH_magnetic_field_Nanoribbon import get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC
from ProjectSpecific.NH_magnetic_field_Nanoribbon import type_2_nonhermitian_magnetic_field_generation_based_nanoribbon
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Get honeycomb nanoribbon plotting info and r values for the sites
sites, conns = plot_honeycomb_nanoribbon(6, 11)
r_vals = identify_r_distance_honeycomb_nanoribbon(6, 11)

# Define colors to be used and custom segmented cmap
colors_list = ['purple', 'blue', 'mediumturquoise', 'green', 'red', 'goldenrod']
custom_cmap = LinearSegmentedColormap.from_list('r_vals_cmap', colors_list, N=6)

# Plot sites with coloring corresponding to r value
plt.scatter(sites[:, 0], sites[:, 1], c=r_vals, cmap=custom_cmap, s=150)

# Plot black lines for NN lattice bonds
all_conns_i = np.full((2 * len(conns[0]), 2), None)
all_conns_j = all_conns_i.copy()
all_conns_i[::2, :] = sites[conns[0], :]
all_conns_j[::2, :] = sites[conns[1], :]
plt.plot([all_conns_i[:, 0], all_conns_j[:, 0]], [all_conns_i[:, 1], all_conns_j[:, 1]],
         color='black', zorder=-1, linewidth=1.8)

# Generate haldane term and convert to sublattice basis
haldane_ham = haldane_current_honeycomb_nanoribbon_PBC(6, 11)
asites, bsites = honeycomb_nanoribbon_site_assignment(6, 11)
sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
haldane_ham = haldane_ham[:, sublattice_basis][sublattice_basis, :]

# Swap directions of B-B bonds so now all A-A and all B-B currents are in same direction
halfway = int(haldane_ham.shape[0] / 2)
haldane_ham[halfway:, halfway:] = np.conj(haldane_ham[halfway:, halfway:])

# Revert haldane term back to natural counting basis
haldane_ham = haldane_ham[:, np.argsort(sublattice_basis)][np.argsort(sublattice_basis), :]

# Get info of sites bordering each plaquette, cut out plaquettes that extend over PBC boundary
plaquette_boundary_sites = get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC(6, 11)
plaquette_boundary_sites = plaquette_boundary_sites[:-4]


# helper function that, for a plaquette, returns NNN hopping sites and the sign of the current
def ccw_currents(single_plaquette_bound_sites, haldane_mat):
    isites = single_plaquette_bound_sites
    jsites = np.roll(single_plaquette_bound_sites, -2)
    hop_vals = np.sign(np.imag(haldane_mat[isites, jsites]))
    return np.hstack((isites.reshape(-1, 1), jsites.reshape(-1, 1), hop_vals.reshape(-1, 1))).astype(np.int64)


# compile all NNN hopping pairs and their sign of current
haldane_currents = np.zeros((1, 3), dtype=np.int64)
for row in range(plaquette_boundary_sites.shape[0]):
    new_currents = ccw_currents(plaquette_boundary_sites[row, :], haldane_ham)
    haldane_currents = np.vstack((haldane_currents,
                                  new_currents))
haldane_currents = haldane_currents[1:, :]

# Define colors to use for NNN current lines
# (first four of prior color list, largest 2 r values do not have associated current)
haldane_color_list = np.array(['purple', 'blue', 'mediumturquoise', 'green'])

# Get the chi(r^2) matrix appearing inside the expm
type2chi = type_2_nonhermitian_magnetic_field_generation_based_nanoribbon(6, 11, 0.001, return_chi_term=True)

# Get chi(r^2) affiliated values with each corresponding color in haldane_color_list
haldane_color_associated_hopping_val = np.unique(np.abs(type2chi))

ax = plt.gca()
# iterate over each NNN pair
for i in range(haldane_currents.shape[0]):
    # check expectation that current is all in same direction
    if haldane_currents[i, 2] == 1:

        # with floating point robustness get color associated with NNN pair
        color_ind_match = haldane_color_associated_hopping_val - np.imag(type2chi[haldane_currents[i, 0], haldane_currents[i, 1]])
        color_ind_match = np.abs(color_ind_match) < 0.00001

        # Plot NNN current line with appropriate color
        plt.plot([sites[haldane_currents[i, 0], 0], sites[haldane_currents[i, 1], 0]],
                 [sites[haldane_currents[i, 0], 1], sites[haldane_currents[i, 1], 1]],
                 color=haldane_color_list[color_ind_match][0], zorder=-1, linewidth=1.0)

        # Draw arrow on middle of current line, in direction of the current
        start_site = sites[haldane_currents[i, 0], :]
        end_site = sites[haldane_currents[i, 1], :]
        unit_vec = (end_site - start_site) / np.linalg.norm(end_site - start_site)
        ax.annotate('',
                    xy=(start_site[0] + 0.6 * unit_vec[0], start_site[1] + 0.6 * unit_vec[1]),
                    xytext=(start_site[0], start_site[1]),
                    arrowprops={'arrowstyle': '-|>', 'mutation_scale': 15, 'lw': 0.0,
                                'color': haldane_color_list[color_ind_match][0]},
                    zorder=100
                    )

    else:
        raise ValueError

ax = plt.gca()
ax.set_aspect('equal')
ax.set_axis_off()
plt.colorbar()

# plt.savefig('svg_files/1_ModelDiagram.svg', transparent=True)
plt.show()
