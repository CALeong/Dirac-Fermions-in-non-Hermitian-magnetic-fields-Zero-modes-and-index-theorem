# Import dependencies
import matplotlib.pyplot as plt
from ProjectSpecific.Honeycomb_Nanoribbon import plot_honeycomb_nanoribbon
from ProjectSpecific.NH_magnetic_field_Nanoribbon import *

# Define system parameters
num_edge_sites = 20
num_levels = 89

# Form Hamiltonian and compute eigenvalues and left and right eigenvectors
ham = type_2_nonhermitian_magnetic_field_generation_based_nanoribbon(num_edge_sites, num_levels, 0.001)
eigvals, left_eigvecs, right_eigvecs = scipy.linalg.eig(ham, left=True, right=True)

print(np.max(np.abs(np.imag(eigvals))))

# Sort eigenproducts so they are ordered with respect to real component of eigenspectrum
left_eigvecs = left_eigvecs[:, np.argsort(eigvals)]
right_eigvecs = right_eigvecs[:, np.argsort(eigvals)]
eigvals = np.sort(eigvals)

# Pick out two closter to zero eigenmodes and compute left and right LDOS
halfway = int(len(eigvals) / 2)
print(eigvals[halfway - 2:halfway + 2])
left_ldos = np.sum(np.abs(left_eigvecs[:, halfway - 1:halfway + 1]) ** 2, axis=1)
right_ldos = np.sum(np.abs(right_eigvecs[:, halfway - 1:halfway + 1]) ** 2, axis=1)

# Get plotting info and process Lattice connections to make it simple to plot below
sites, conns = plot_honeycomb_nanoribbon(num_edge_sites, num_levels)
sites_conns_i = np.full((2 * len(conns[0]), 2), None, dtype=np.float64)
sites_conns_j = np.full((2 * len(conns[0]), 2), None, dtype=np.float64)
sites_conns_i[0::2] = sites[conns[0], :]
sites_conns_j[0::2] = sites[conns[1], :]

# Define global plotting parameters
scatter_size = 20

# f1 = left_ldos / np.max(left_ldos)
# f2 = right_ldos / np.max(right_ldos)
# diff = np.abs(f1 - f2)
# plt.imshow(diff.reshape(89, 20))
# plt.show()

# Plot lattice and color sites with left eigenvector LDOS
# Order layers of sites plotted such that higher LDOS are on high layer so they can be seen more easily
# print(np.min(np.sort(left_ldos) / np.max(left_ldos)))
print((left_ldos / np.max(left_ldos))[:20], (left_ldos / np.max(left_ldos))[-20:])
cmap_ordering = np.argsort(left_ldos)
plt.scatter(sites[cmap_ordering, 0], sites[cmap_ordering, 1],
            c=(np.sort(left_ldos) / np.max(left_ldos)), cmap='inferno', s=scatter_size)
plt.plot([sites_conns_i[:, 0], sites_conns_j[:, 0]],
         [sites_conns_i[:, 1], sites_conns_j[:, 1]], color='black', zorder=-1)
ax = plt.gca()
ax.set_axis_off()
ax.set_aspect('equal')
plt.colorbar()
# plt.savefig('svg_files/type2_left_eigenvector_LDOS.svg')
plt.show()

# Same as above but for right eigenvectors LDOS
# print(np.min(np.sort(right_ldos) / np.max(right_ldos)))
print((right_ldos / np.max(right_ldos))[:20], (right_ldos / np.max(right_ldos))[-20:])
cmap_ordering = np.argsort(right_ldos)
plt.scatter(sites[cmap_ordering, 0], sites[cmap_ordering, 1],
            c=(np.sort(right_ldos) / np.max(right_ldos)), cmap='inferno', s=scatter_size)
plt.plot([sites_conns_i[:, 0], sites_conns_j[:, 0]],
         [sites_conns_i[:, 1], sites_conns_j[:, 1]], color='black', zorder=-1)
ax = plt.gca()
ax.set_axis_off()
ax.set_aspect('equal')
plt.colorbar()
# plt.savefig('svg_files/type2_right_eigenvector_LDOS.svg')
plt.show()
