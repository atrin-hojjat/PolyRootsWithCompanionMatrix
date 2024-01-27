import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_gershgorin_discs(matrix):
    n = len(matrix)
    eigenvalues, _ = np.linalg.eig(matrix)

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')

    for i in range(n):
        disc_center = matrix[i, i]
        disc_radius = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])

        disc = plt.Circle((disc_center, 0), disc_radius, fill=False, color='b', linestyle='dashed')
        ax.add_patch(disc)
        ax.plot(disc_center, 0, 'bo')  # Plot the center of the disc

    ax.set_title('Gershgorin Discs')
    ax.grid(True)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')

    ax.plot(np.real(eigenvalues), np.imag(eigenvalues), 'ro', label='Eigenvalues')
    plt.legend()

    plt.savefig(os.path.join("../output", "gershgorin.jpg"))
    matplotlib.rcParams.update({
        "pgf.texsystem": "xelatex",
        'text.usetex': True,
        'pgf.rcfonts': False,
        "font.family": "mononoki Nerd Font Mono",
        "font.serif": [],
        #  "font.cursive": ["mononoki Nerd Font", "mononoki Nerd Font Mono"],
    })
    plt.savefig(os.path.join("../output", "gershgorin.pgf"))

    plt.show()


def get_companion_matrix(poly):
    """
    Calculate the companion matrix for a normalized polynomial
    """
    n = len(poly)
    cmat = np.zeros((n - 1, n - 1))
    cmat[:, n - 2] = (-poly[1:])[::-1]
    cmat[np.arange(1, n - 1), np.arange(0, n - 2)] = 1

    return cmat


def get_formated_poly(poly):
    """
    Return the polynomial such that the coefficient of the maximum power of x 
    is always 1
    """
    return poly / poly[0]


poly = np.array([2, 3, 1, 4, 3, 6])

norm_poly = get_formated_poly(poly)
matrix = get_companion_matrix(norm_poly)

plot_gershgorin_discs(matrix)
