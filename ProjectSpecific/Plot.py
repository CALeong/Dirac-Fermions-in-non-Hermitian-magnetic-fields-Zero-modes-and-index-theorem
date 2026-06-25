import numpy as np
import scipy
from Lattice.Bernal_Bilayer_Honeycomb_Sparse import honeycomb_flake_plot


def plot_zero_energy_ldos(nval, ham_sublattice_basis, subtract_zero_field=False):
    eigvals, left_eigvecs, right_eigvecs = scipy.linalg.eig(ham_sublattice_basis, left=True, right=True)

    print('#################### Checks ####################')
    print('Eigenspectrum max imaginary component: ', np.max(np.imag(eigvals)))

    zero_energy_location = (np.abs(eigvals) - np.min(np.abs(eigvals))) < 0.001
    print('Number of zero modes: ', np.sum(zero_energy_location))

    left_zero_eigvecs = left_eigvecs[:, zero_energy_location]
    right_zero_eigvecs = right_eigvecs[:, zero_energy_location]

    left_zero_ldos = np.sum(np.abs(left_zero_eigvecs) ** 2, axis=1)
    left_zero_ldos = left_zero_ldos / np.sum(left_zero_ldos)
    right_zero_ldos = np.sum(np.abs(right_zero_eigvecs) ** 2, axis=1)
    right_zero_ldos = right_zero_ldos / np.sum(right_zero_ldos)


