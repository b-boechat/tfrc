def calculate_gini_index(arr):
    # Calcula o coeficiente Gini para os elementos de um array arr.
    arr_sorted = np.sort(arr, axis=None)
    N = len(arr_sorted)
    return 1 - 2 * np.sum(np.multiply(arr_sorted, (N - np.arange(1, N + 1) + 1 / 2) / N)) / (sum(arr_sorted) + epsilon)


def calculate_energy(arr):
    lines, columns = arr.shape
    window = generate_asym_hamming(lines, columns)
    return np.sum(np.multiply(arr, window))


def generate_asym_hamming(lines, columns):
    hamm = np.outer(np.hamming(lines), np.hamming(columns * 2 - 1))
    return hamm[:, 0:columns]


def local_sparsity_method(X, analysis_width=3, analysis_height=5):
    P, K, M = X.shape

    P_tilde = np.zeros((K, M))

    for k in range(K):
        for m in range(M):
            max_gini = -1
            min_energy = np.inf
            for p in range(P):
                gini = calculate_gini_index(X[p, max(0, k - analysis_height):min(K, k + analysis_height + 1),
                                            max(0, m - analysis_width):min(M, m + analysis_width + 1)])
                energy = calculate_energy(
                    X[p, max(0, k - analysis_height):min(K, k + analysis_height + 1), max(0, m - analysis_width):m + 1])
                if energy < min_energy:
                    min_energy = energy

                if gini >= max_gini:
                    max_gini = gini
                    P_tilde[k, m] = X[p, k, m] / energy

            P_tilde[k, m] *= min_energy

    return P_tilde