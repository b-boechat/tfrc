# Versão do Fast Local Sparsity que realiza combinação por média geométrica nas regiões de menor energia, para diminuir o custo computacional. No momento, energia L2 ainda é calculada em todo o tensor.
# Ainda está sendo implementada.

import numpy as np
from scipy.signal import correlate
from libc.math cimport pow, sqrt, INFINITY, log10
cimport cython

DEF DEBUGPRINT = 0
DEF DEBUGTIMER = 0
DEF DEBUGHISTOGRAM = 0

IF DEBUGHISTOGRAM:
    import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

IF DEBUGPRINT:
    import colorama
    from debug import print_arr


IF DEBUGTIMER:
    from libc.time cimport clock_t, clock, CLOCKS_PER_SEC


def fast_local_sparsity_hybrid_wrapper(X, freq_width=39, time_width=11, eta=8.0, energy_criterium_db=-40):
    #print(f"freq_width_sparsity = {freq_width_sparsity}\nfreq_width_energy = {freq_width_energy}\ntime_width = {time_width}\nzeta = {zeta}")
    return fast_local_sparsity_hybrid(X, freq_width, time_width, eta, energy_criterium_db)

#@cython.boundscheck(False)
#@cython.wraparound(False) 
#@cython.nonecheck(False)
#@cython.cdivision(True)
cdef fast_local_sparsity_hybrid(double[:,:,::1] X, Py_ssize_t freq_width, Py_ssize_t time_width, double eta, double energy_criterium_db):

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

    cdef double max_local_energy_db

    local_energy_l1_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] local_energy_l1
    local_energy_l2_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] local_energy_l2

    IF DEBUGTIMER:
        cdef clock_t time_i, time_f
    

    #cdef Py_ssize_t count1 = 0, count2 = 0


    ############ Cálculo da suitability local (por esparsidade) {{{

    IF DEBUGTIMER:
        time_i = clock() 

    for p in range(P):
        # Calcula a matriz de energias local de normas L1 do espectrograma.
        local_energy_l1_ndarray[p] = correlate(X_ndarray[p], hamming_window, mode='same') + epsilon
        local_energy_l2_ndarray[p] = np.sqrt(correlate(X_ndarray[p] * X_ndarray[p], hamming_window * hamming_window, mode='same') + epsilon)

    IF DEBUGTIMER:
        time_f = clock()
        print(f"Timer: {<double> (time_f - time_i) / CLOCKS_PER_SEC}") 

    local_energy_l1 = local_energy_l1_ndarray
    local_energy_l2 = local_energy_l2_ndarray
 
    IF DEBUGHISTOGRAM:
        plt.figure()
        plt.hist(10*np.log10(local_energy_l1_ndarray[0].flatten()), bins=50)

        plt.figure()
        plt.hist(10*np.log10(local_energy_l1_ndarray[1].flatten()), bins=50)

        plt.figure()
        plt.hist(10*np.log10(local_energy_l1_ndarray[2].flatten()), bins=50)
        plt.show()

    for k in range(K):
        for m in range(M):
            # Encontra a maior energia local.
            max_local_energy_db = -INFINITY
            for p in range(P):
                if 10*log10(local_energy_l1[p, k, m]) > max_local_energy_db:
                    max_local_energy_db = 10*log10(local_energy_l1[p, k, m])
            
            # Se essa energia está abaixo do critério escolhido, realiza combinação por média geométrica.
            if max_local_energy_db < energy_criterium_db:
                #count1 = count1 + 1
                result[k, m] = 1.0
                for p in range(P):
                    result[k, m] = result[k, m] * X[p, k, m]
                result[k, m] = pow(result[k, m], 1.0/P)

            # Caso contrário, realiza a combinação FLS.
            else:
                #count2 = count2 + 1
                for p in range(P):
                    suitability[p, k, m] = (window_size_sqrt - local_energy_l1[p, k, m]/local_energy_l2[p, k, m])/ \
                                        ((window_size_sqrt - 1) * sqrt(local_energy_l1[p, k, m])) + epsilon
                suitability_product = 1
                weights_sum = 0
                for p in range(P):
                    suitability_product = suitability_product * suitability[p, k, m]
                for p in range(P):
                    weight = pow(suitability[p, k, m] * suitability[p, k, m] / suitability_product, eta)
                    weights_sum = weights_sum + weight
                    result[k, m] = result[k, m] + weight * X[p, k, m]
                result[k, m] = result[k, m] / weights_sum

    #print(count1, count2)

    return result_ndarray








        
