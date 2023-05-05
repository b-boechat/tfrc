import numpy as np
from scipy.signal import correlate
from libc.math cimport pow, exp, sqrt
cimport cython


DEF DEBUGPRINT = 0

IF DEBUGPRINT:
    import colorama
    from debug import print_arr

#IF DEBUGTIMER:
#    from libc.time cimport clock_t, clock

def fast_local_sparsity_wrapper(X, freq_width=39, time_width=11, eta=8.0):
    return fast_local_sparsity(X, freq_width, time_width, eta)

def fast_local_sparsity_opt_wrapper(X, freq_width=39, time_width=11, eta=8.0):
    return fast_local_sparsity_opt(X, freq_width, time_width, eta)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef fast_local_sparsity(double[:,:,::1] X, Py_ssize_t freq_width, Py_ssize_t time_width, double eta):

    cdef:
        Py_ssize_t P = X.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X.shape[1] # Eixo das frequências
        Py_ssize_t M = X.shape[2] # Eixo do tempo

        double epsilon = 1e-10

        double window_size_sqrt = sqrt(<double> freq_width * time_width)
        double suitability_product, weight, weights_sum

    X_ndarray = np.asarray(X)

    # Container que armazena o resultado.
    result_ndarray = np.zeros((K, M), dtype=np.double)
    cdef double[:,:] result = result_ndarray

    # Gera a janela de Hamming 2D com as dimensões fornecidas.
    hamming_window = np.outer(np.hamming(freq_width), np.hamming(time_width))

    # Cointainer que armazena a "suitability" da região do espectrograma, calculada a partir da medida de esparsidade local.
    suitability_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] suitability = suitability_ndarray

    cdef: 
        double[:,:] local_energy_l1
        double[:,:] local_energy_l2
        double[:,:] local_energy_l1_sqrt

    ############ Cálculo da suitability local (por esparsidade) {{{

    for p in range(P):
        # Calcula as matrizes de energias locais de normas L1 e L2 do espectrograma, bem como a raiz quadrada element-wise da matriz L1.
        local_energy_l1_ndarray = correlate(X_ndarray[p], hamming_window, mode='same') + epsilon
        local_energy_l2_ndarray = np.sqrt(correlate(X_ndarray[p] * X_ndarray[p], hamming_window * hamming_window, mode='same') + epsilon)
        local_energy_l1_sqrt_ndarray = np.sqrt(local_energy_l1_ndarray)

        # Aponta os memviews de Cython para as matrizes calculadas.
        local_energy_l1 = local_energy_l1_ndarray
        local_energy_l2 = local_energy_l2_ndarray
        local_energy_l1_sqrt = local_energy_l1_sqrt_ndarray

        # Calcula a "suitability" da forma descrita, a partir da medida de esparsidade de Hoyer
        for k in range(K):
            for m in range(M):
                suitability[p, k, m] = (window_size_sqrt - local_energy_l1[k, m]/local_energy_l2[k, m])/ \
                                        ((window_size_sqrt - 1) * local_energy_l1_sqrt[k, m]) + epsilon


    IF DEBUGPRINT:
        print("Suitability")
        for p in range(P):
            print(f"p = {p}")
            print_arr(suitability[p])

    ############ Cálculo da suitability local (por esparsidade) }}}

    ############ Combinação dos espectrogramas {{{
    
    for k in range(K):
        for m in range(M):
            suitability_product = 1
            weights_sum = 0
            for p in range(P):
                suitability_product = suitability_product * suitability[p, k, m]
            for p in range(P):
                weight = pow(suitability[p, k, m] * suitability[p, k, m] / suitability_product, eta)
                weights_sum = weights_sum + weight

                result[k, m] = result[k, m] + weight * X[p, k, m]
            result[k, m] = result[k, m] / weights_sum

    ############ Combinação dos espectrogramas }}}

    return result_ndarray



@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef fast_local_sparsity_opt(double[:,:,::1] X, Py_ssize_t freq_width, Py_ssize_t time_width, double eta):

    cdef:
        Py_ssize_t P = X.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X.shape[1] # Eixo das frequências
        Py_ssize_t M = X.shape[2] # Eixo do tempo

        double epsilon = 1e-10

        double window_size_sqrt = sqrt(<double> freq_width * time_width)

    X_ndarray = np.asarray(X)

    # Container que armazena o resultado.
    result_ndarray = np.zeros((K, M), dtype=np.double)
    cdef double[:,:] result = result_ndarray

    # Gera a janela de Hamming 2D com as dimensões fornecidas.
    hamming_window = np.outer(np.hamming(freq_width), np.hamming(time_width))

    # Cointainer que armazena a "suitability" da região do espectrograma, calculada a partir da medida de esparsidade local.
    suitability_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] suitability = suitability_ndarray

    # Variáveis referentes à combinação.
    cdef double[:, :, :] log_suitability
    cdef double[:, :] sum_log_suitability
    combination_weight_ndarray = np.empty((P, K, M), dtype=np.double)
    cdef double[:, :, :] combination_weight = combination_weight_ndarray

    cdef: 
        double[:,:] local_energy_l1
        double[:,:] local_energy_l2
        double[:,:] local_energy_l1_sqrt

    ############ Cálculo da suitability local (por esparsidade) {{{

    for p in range(P):
        # Calcula as matrizes de energias locais de normas L1 e L2 do espectrograma, bem como a raiz quadrada element-wise da matriz L1.
        local_energy_l1_ndarray = correlate(X_ndarray[p], hamming_window, mode='same') + epsilon
        local_energy_l2_ndarray = np.sqrt(correlate(X_ndarray[p] * X_ndarray[p], hamming_window * hamming_window, mode='same') + epsilon)
        local_energy_l1_sqrt_ndarray = np.sqrt(local_energy_l1_ndarray)

        # Aponta os memviews de Cython para as matrizes calculadas.
        local_energy_l1 = local_energy_l1_ndarray
        local_energy_l2 = local_energy_l2_ndarray
        local_energy_l1_sqrt = local_energy_l1_sqrt_ndarray

        # Calcula a "suitability" da forma descrita, a partir da medida de esparsidade de Hoyer
        for k in range(K):
            for m in range(M):
                suitability[p, k, m] = (window_size_sqrt - local_energy_l1[k, m]/local_energy_l2[k, m])/ \
                                        ((window_size_sqrt - 1) * local_energy_l1_sqrt[k, m]) + epsilon


    IF DEBUGPRINT:
        print("Suitability")
        for p in range(P):
            print(f"p = {p}")
            print_arr(suitability[p])

    ############ Cálculo da suitability local (por esparsidade) }}}

    ############ Combinação dos espectrogramas {{{

    log_suitability_ndarray = np.log(suitability_ndarray)
    sum_log_suitability_ndarray = np.sum(log_suitability_ndarray, axis=0)

    log_suitability = log_suitability_ndarray
    sum_log_suitability = sum_log_suitability_ndarray

    for p in range(P):
        for k in range(K): 
            for m in range(M):
                combination_weight[p, k, m] = exp( (2*log_suitability[p, k, m] - sum_log_suitability[k, m]) * eta)

    result_ndarray = np.average(X_ndarray, axis=0, weights=combination_weight_ndarray)

    ############ Combinação dos espectrogramas }}}

    return result_ndarray








        
