import numpy as np
from timeit import default_timer

from numpy.core.fromnumeric import shape # DEBUG APENAS

# IMPLEMENTAÇÃO ANTIGA

def calculate_gini_index(arr):
    epsilon = 0.00001

    lines, columns = arr.shape

    window = generate_hamming(lines, columns)

    # Calcula o coeficiente Gini para os elementos de um array arr.
    arr_sorted = np.sort(np.multiply(arr, window), axis=None)
    N = len(arr_sorted)
    return 1 - 2 * np.sum(np.multiply(arr_sorted, (N - np.arange(1, N + 1) + 1 / 2) / N)) / (sum(arr_sorted) + epsilon)


def calculate_energy(arr):
    lines, columns = arr.shape
    window = generate_asym_hamming(lines, columns)
    return np.sum(np.multiply(arr, window))

def generate_hamming(lines, columns):
    hamm = np.outer(np.hamming(lines), np.hamming(columns))
    return hamm

def generate_asym_hamming(lines, columns):
    hamm = np.outer(np.hamming(lines), np.hamming(columns * 2 - 1))
    return hamm[:, 0:columns]


def local_sparsity_method_python(X, analysis_width=3, analysis_height=5):
    P, K, M = X.shape

    P_tilde = np.zeros((K, M))
    time_init = default_timer()

    sparsity_timer = 0
    energy_timer = 0 

    for k in range(K):
        for m in range(M):
            max_gini = -1
            min_energy = np.inf
            for p in range(P):
                time_i = default_timer()
                gini = calculate_gini_index(X[p, max(0, k - analysis_height):min(K, k + analysis_height + 1),
                                            max(0, m - analysis_width):min(M, m + analysis_width + 1)])
                time_f = default_timer()
                sparsity_timer += time_f - time_i


                time_i_energy = default_timer()
                energy = calculate_energy(
                    X[p, max(0, k - analysis_height):min(K, k + analysis_height + 1), max(0, m - analysis_width):m + 1])
                time_f_energy = default_timer()
                energy_timer += time_f_energy - time_i_energy
                if energy < min_energy:
                    min_energy = energy

                if gini >= max_gini:
                    max_gini = gini
                    P_tilde[k, m] = X[p, k, m] / energy

            P_tilde[k, m] *= min_energy

    time_final = default_timer()

    print(f"\tSparsity = {sparsity_timer}\n\tEnergy = {energy_timer}\n\tTotal = {time_final - time_init}")

    return P_tilde