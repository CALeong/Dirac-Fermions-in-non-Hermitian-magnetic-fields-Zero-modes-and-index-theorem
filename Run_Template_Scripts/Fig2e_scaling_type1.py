# Import dependencies
import numpy as np
from scipy.optimize import curve_fit
from ProjectSpecific.NH_magnetic_field_Nanoribbon import type_1_nonhermitian_magnetic_field_nanoribbon
from ProjectSpecific.NH_magnetic_field_Nanoribbon import get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC
import matplotlib.pyplot as plt


# Function for linear fit later
def linear_fit(x, m):
    return m * x


# Define number of levels of the nanoribbon to sample
num_levels_list = np.arange(19, 100, 10).astype(np.int64)

# Define zero window boundary
zero_window_bound = 0.125

# Define variables to hold final values
num_zero_modes = np.array([])
num_plaquettes = np.array([])
max_imag = np.array([])

# Iterate over all system sizes
for nl in num_levels_list:
    # For type1 Hamiltonian with proper size and get eigenvalues
    ham = type_1_nonhermitian_magnetic_field_nanoribbon(20, nl, 0.01)
    eigvals = np.linalg.eigvals(ham)

    # Append to data arrays relevant values
    num_zero_modes = np.append(num_zero_modes, np.sum(np.abs(np.real(eigvals)) <= zero_window_bound))

    num_plaquettes = np.append(num_plaquettes,
                               get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC(20, nl).shape[0])

    max_imag = np.append(max_imag, np.max(np.abs(np.imag(eigvals))))

# Perform fit to get slope and error
fit, error = curve_fit(linear_fit, num_plaquettes, num_zero_modes)  # noqa
print(fit, error)

print(np.max(max_imag))

# Plot things
plt.scatter(num_plaquettes, num_zero_modes, color='black', s=80)
plt.plot(np.linspace(0, 1000, 2), linear_fit(np.linspace(0, 1000, 2), fit[0]), color='red', lw=2, linestyle='--', zorder=-1)
plt.xlim([0, 1000])
plt.ylim([0, 10.5])
plt.tick_params(axis='both', direction='in')
plt.xticks([0, 500, 1000])
plt.yticks([0, 5, 10])
ax = plt.gca()
ax.set_box_aspect(0.5)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
# plt.savefig('svg_files/type1_zero_modes_scaling_honeycomb_nanoribbon.svg')
plt.show()


