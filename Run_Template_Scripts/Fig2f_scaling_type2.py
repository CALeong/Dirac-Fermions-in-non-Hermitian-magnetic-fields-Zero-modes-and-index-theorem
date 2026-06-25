import numpy as np
from scipy.optimize import curve_fit
from ProjectSpecific.NH_magnetic_field_Nanoribbon import type_2_nonhermitian_magnetic_field_generation_based_nanoribbon
from ProjectSpecific.NH_magnetic_field_Nanoribbon import get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC
import matplotlib.pyplot as plt


def linear_fit(x, m):
    return m * x


num_levels_list = np.arange(19, 100, 10).astype(np.int64)
zero_window_bound = 0.125

num_zero_modes = np.array([])
num_plaquettes = np.array([])
max_imag = np.array([])
for nl in num_levels_list:
    ham = type_2_nonhermitian_magnetic_field_generation_based_nanoribbon(20, nl, 0.001)
    eigvals = np.linalg.eigvals(ham)

    num_zero_modes = np.append(num_zero_modes, np.sum(np.abs(np.real(eigvals)) <= zero_window_bound))

    num_plaquettes = np.append(num_plaquettes,
                               get_plaquette_boundary_sites_honeycomb_nanoribbon_PBC(20, nl).shape[0])

    max_imag = np.append(max_imag, np.max(np.abs(np.imag(eigvals))))

fit, error = curve_fit(linear_fit, num_plaquettes, num_zero_modes)  # noqa
print(fit, error)

print(np.max(max_imag))

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
# plt.savefig('svg_files/type2_zero_modes_scaling_honeycomb_nanoribbon.svg')
plt.show()


