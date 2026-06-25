# Import dependencies
import numpy as np
import scipy
from ProjectSpecific.Honeycomb_Nanoribbon import plot_honeycomb_nanoribbon
from ProjectSpecific.NH_magnetic_field_Nanoribbon import type_1_nonhermitian_magnetic_field_nanoribbon
import matplotlib.pyplot as plt

# Generate Hamiltonian and get eigenvalues and left and right eigenvectors
ham = type_1_nonhermitian_magnetic_field_nanoribbon(20, 89, 0.01)
eigvals, left_eigvecs, right_eigvecs = scipy.linalg.eig(ham, left=True, right=True)
print(np.max(np.abs(np.imag(eigvals))))

# Order eigenproducts such that they are ordered with respect to real part of eigenspectrum
left_eigvecs = left_eigvecs[:, np.argsort(eigvals)]
right_eigvecs = right_eigvecs[:, np.argsort(eigvals)]
eigvals = np.sort(eigvals)

# Take two closest to zero modes and compute LDOS over each site
halfway = int(ham.shape[0] / 2)
zero_inds = [halfway - 1, halfway]
print(eigvals[halfway - 2:halfway + 2])
left_ldos_zero_energy = np.sum(np.abs(left_eigvecs[:, zero_inds]) ** 2, axis=1)
right_ldos_zero_energy = np.sum(np.abs(right_eigvecs[:, zero_inds]) ** 2, axis=1)

# get plotting information and process lattice NN bonds info to easily plottable form
sites, conns = plot_honeycomb_nanoribbon(20, 89)
sites_conns_i = np.full((2 * len(conns[0]), 2), None)
sites_conns_i[::2] = sites[conns[0], :]
sites_conns_j = np.full((2 * len(conns[1]), 2), None)
sites_conns_j[::2] = sites[conns[1], :]

# Plot left eigenvector LDOS on nanoribbon lattice with cmap='inferno'
# Sort points such that larger LDOS are plotted on higher layer so they can be seen more easily
# print(np.min(np.sort(left_ldos_zero_energy) / np.max(left_ldos_zero_energy)))
cmap_magnitude_ordering = np.argsort(left_ldos_zero_energy)
plt.scatter(sites[cmap_magnitude_ordering, 0], sites[cmap_magnitude_ordering, 1],
            c=(np.sort(left_ldos_zero_energy) / np.max(left_ldos_zero_energy)), cmap='inferno', s=20)
plt.plot([sites_conns_i[:, 0], sites_conns_j[:, 0]], [sites_conns_i[:, 1], sites_conns_j[:, 1]],
         color='black', zorder=-1)
ax = plt.gca()
ax.set_axis_off()
ax.set_aspect('equal')
plt.colorbar()
# plt.savefig('svg_files/type1_left_eigenvector_LDOS.svg', transparent=True)
plt.show()

# Same as above but for right eigenvectors
# print(np.min(np.sort(right_ldos_zero_energy) / np.max(right_ldos_zero_energy)))
cmap_magnitude_ordering = np.argsort(right_ldos_zero_energy)
plt.scatter(sites[cmap_magnitude_ordering, 0], sites[cmap_magnitude_ordering, 1],
            c=(np.sort(right_ldos_zero_energy) / np.max(right_ldos_zero_energy)), cmap='inferno', s=20)
plt.plot([sites_conns_i[:, 0], sites_conns_j[:, 0]], [sites_conns_i[:, 1], sites_conns_j[:, 1]],
         color='black', zorder=-1)
ax = plt.gca()
ax.set_axis_off()
ax.set_aspect('equal')
plt.colorbar()
# plt.savefig('svg_files/type1_right_eigenvector_LDOS.svg', transparent=True)
plt.show()

