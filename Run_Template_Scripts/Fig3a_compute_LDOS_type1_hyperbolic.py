import scipy
import numpy as np
from ProjectSpecific.NH_magnetic_field_hyperbolic import type1_nonhermitian_uniform_magnetic_field_hyperbolic
from Lattice.General_Hamiltonian import first_site_on_gen_hyperbolic_q3
import matplotlib.pyplot as plt


hamiltonian = type1_nonhermitian_uniform_magnetic_field_hyperbolic(10, 4, 1.0)

eigvals, left_eigvecs, right_eigvecs = scipy.linalg.eig(hamiltonian, left=True, right=True)

left_eigvecs = left_eigvecs[:, np.argsort(eigvals)]
right_eigvecs = right_eigvecs[:, np.argsort(eigvals)]
eigvals = np.sort(eigvals)

print(np.max(np.abs(np.imag(eigvals))))

halfway = int(hamiltonian.shape[0] / 2)
print(eigvals[halfway - 3:halfway + 3])

left_ldos = np.sum(np.abs(left_eigvecs[:, halfway - 2:halfway + 2]) ** 2, axis=1)
right_ldos = np.sum(np.abs(right_eigvecs[:, halfway - 2:halfway + 2]) ** 2, axis=1)

left_ldos = left_ldos / np.max(left_ldos)
right_ldos = right_ldos / np.max(right_ldos)

first_site_on_gen = first_site_on_gen_hyperbolic_q3(10, 4)
left_gen_avg_ldos = np.array([])
right_gen_avg_ldos = np.array([])
for n in range(4):
    sites_on_gen = np.arange(first_site_on_gen[n], first_site_on_gen[n + 1]).astype(np.int64)
    left_gen_avg_ldos = np.append(left_gen_avg_ldos, np.average(left_ldos[sites_on_gen]))
    right_gen_avg_ldos = np.append(right_gen_avg_ldos, np.average(right_ldos[sites_on_gen]))

fig = plt.figure()
ax_left = plt.gca()
ax_left.set_xticks([1, 2, 3, 4])
ax_left.scatter(range(1, 5), left_gen_avg_ldos, s=100, marker='^', facecolors='none', edgecolors='red', linewidths=2)
ax_left.plot(range(1, 5), left_gen_avg_ldos, lw=2.0, linestyle='--', color='red', zorder=-1)
ax_right = ax_left.twinx()
ax_right.scatter(range(1, 5), right_gen_avg_ldos, s=100, marker='v', facecolors='none', edgecolors='blue', linewidths=2)
ax_right.plot(range(1, 5), right_gen_avg_ldos, lw=2.0, linestyle='--', color='blue', zorder=-1)
ax_left.set_yticks([0.0, 0.3, 0.6])
ax_right.set_yticks([0.00, 0.03, 0.06])
ax_left.set_xlim(0.9, 4.1)
ax_left.set_ylim(-0.05, 0.65)
ax_right.set_ylim(-0.005, 0.065)
ax_left.set_box_aspect(0.5)
ax_right.set_box_aspect(0.5)
ax_left.spines['left'].set_linewidth(2)
ax_right.spines['right'].set_linewidth(2)
ax_left.spines['top'].set_linewidth(2)
ax_left.spines['bottom'].set_linewidth(2)
ax_left.spines['left'].set_color('red')
ax_left.spines['right'].set_color('blue')
ax_right.spines['left'].set_color('red')
ax_right.spines['right'].set_color('blue')
ax_left.tick_params(axis='both', direction='in')
ax_right.tick_params(axis='both', direction='in')
# plt.savefig('svg_files/type1_ldos_hyperbolic_p10q3n4.svg')
plt.show()



