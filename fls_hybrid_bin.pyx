# Versão do Fast Local Sparsity que realiza combinação por média geométrica nos bins de menor energia, para diminuir o custo computacional. Em contraste com o fls_hybrid, essa utiliza um critério binwise de energia.
# Ainda sendo implementada. No momento, só o que está implementado além do fls_hybrid é o cálculo "manual" das energias L1 e L2 para comparar com o uso do scipy correlate.

import numpy as np
from scipy.signal import correlate
from libc.math cimport pow, sqrt, INFINITY, log10
cimport cython

DEF DEBUGPRINT = 0
DEF DEBUGTIMER = 0
DEF DEBUGHISTOGRAM = 0

IF DEBUGHISTOGRAM:
    import matplotlib.pyplot as plt

IF DEBUGPRINT:
    import colorama
    from debug import print_arr


IF DEBUGTIMER:
    from libc.time cimport clock_t, clock, CLOCKS_PER_SEC


def fast_local_sparsity_hybrid_bin_wrapper(X, freq_width=39, time_width=11, eta=8.0, energy_criterium_db=-40):
    #print(f"freq_width_sparsity = {freq_width_sparsity}\nfreq_width_energy = {freq_width_energy}\ntime_width = {time_width}\nzeta = {zeta}")
    return fast_local_sparsity_hybrid_bin(X, freq_width, time_width, eta, energy_criterium_db)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef fast_local_sparsity_hybrid_bin(double[:,:,::1] X_orig, Py_ssize_t freq_width, Py_ssize_t time_width, double eta, double energy_criterium_db):

    cdef:
        Py_ssize_t P = X_orig.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X_orig.shape[1] # Eixo das frequências
        Py_ssize_t M = X_orig.shape[2] # Eixo do tempo

        cdef Py_ssize_t p, k, m, i, j

        double epsilon = 1e-10
        Py_ssize_t freq_width_lobe = (freq_width-1)//2
        Py_ssize_t time_width_lobe = (time_width-1)//2

        double window_size_sqrt = sqrt(<double> freq_width * time_width)
        double suitability_product, weight, weights_sum

    # Realiza zero-padding no tensor de espectrogramas.
    X_ndarray = np.pad(X_orig, ((0, 0), (freq_width_lobe, freq_width_lobe), (time_width_lobe, time_width_lobe)))
    cdef double[:, :, :] X = X_ndarray

    # Temp.
    X_orig_ndarray = np.asarray(X_orig)

    # Container que armazena o resultado.
    result_ndarray = np.zeros((K, M), dtype=np.double)
    cdef double[:,:] result = result_ndarray

    # Gera as janelas de Hamming com as dimensões adequadas em cada eixo.
    #hamming_freq_ndarray = np.hamming(freq_width)
    #hamming_time_ndarray = np.hamming(time_width)
    #cdef double[:] hamming_freq = hamming_freq_ndarray
    #cdef double[:] hamming_time = hamming_time_ndarray

    # TEMP
    hamming_window_ndarray = np.outer(np.hamming(freq_width), np.hamming(time_width))
    cdef double[:, :] hamming_window = hamming_window_ndarray
    
    # Cointainer que armazena a "suitability" da região do espectrograma, calculada a partir da medida de esparsidade local.
    suitability_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] suitability = suitability_ndarray

    cdef double max_local_energy_db

    local_energy_l1_ndarray = epsilon * np.ones((P, K, M), dtype=np.double)
    cdef double[:,:,:] local_energy_l1 = local_energy_l1_ndarray
    local_energy_l2_ndarray = epsilon * np.ones((P, K, M), dtype=np.double)
    cdef double[:,:,:] local_energy_l2 = local_energy_l2_ndarray

    IF DEBUGTIMER:
        cdef clock_t time_i, time_f
    

    #cdef Py_ssize_t count1 = 0, count2 = 0


    for p in range(P):
        for k in range(freq_width_lobe, K + freq_width_lobe):
           for m in range(time_width_lobe, M + time_width_lobe):
                # Calcula as energia locais no segmento definido por (p, k, m).     
                for i in range(freq_width):
                   for j in range(time_width):
                       local_energy_l1[p, k - freq_width_lobe, m - time_width_lobe] += X[p, k - freq_width_lobe + i, m - time_width_lobe + j] * hamming_window[i, j]
                       local_energy_l2[p, k - freq_width_lobe, m - time_width_lobe] += X[p, k - freq_width_lobe + i, m - time_width_lobe + j] * X[p, k - freq_width_lobe + i, m - time_width_lobe + j] * hamming_window[i, j] * hamming_window[i, j]
                local_energy_l2[p, k - freq_width_lobe, m - time_width_lobe] = sqrt(local_energy_l2[p, k - freq_width_lobe, m - time_width_lobe])
                    


    ############ Cálculo da suitability local (por esparsidade) {{{

    IF DEBUGTIMER:
        time_i = clock() 

    #for p in range(P):
        # Calcula a matriz de energias local de normas L1 do espectrograma.
        #local_energy_l1_ndarray[p] = correlate(X_ndarray[p], hamming_window, mode='same') + epsilon
        #local_energy_l2_ndarray[p] = np.sqrt(correlate(X_orig_ndarray[p] * X_orig_ndarray[p], hamming_window_ndarray * hamming_window_ndarray, mode='same') + epsilon)

    IF DEBUGTIMER:
        time_f = clock()
        print(f"Timer: {<double> (time_f - time_i) / CLOCKS_PER_SEC}") 

    #local_energy_l1 = local_energy_l1_ndarray
    #local_energy_l2 = local_energy_l2_ndarray
 
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
            #if max_local_energy_db < energy_criterium_db:
            if False: # bypassado por enquanto.
                #count1 = count1 + 1
                result[k, m] = 1.0
                for p in range(P):
                    result[k, m] = result[k, m] * X_orig[p, k, m]
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
                    result[k, m] = result[k, m] + weight * X_orig[p, k, m]
                result[k, m] = result[k, m] / weights_sum

    #print(count1, count2)

    return result_ndarray








        
