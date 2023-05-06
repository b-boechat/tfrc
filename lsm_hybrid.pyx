# Versão do Local Sparsity que realiza combinação por média geométrica nas regiões de menor energia, para diminuir o custo computacional.

import numpy as np
from scipy.signal import correlate
cimport cython
from libc.math cimport INFINITY, exp, log10

DEF DEBUGPRINT = 0
DEF DEBUGTIMER = 0
DEF DEBUGHISTOGRAM = 0
DEF DEBUGCOUNT = 0

IF DEBUGPRINT:
    import colorama
    from debug import print_arr

IF DEBUGTIMER:
    from libc.time cimport clock_t, clock

IF DEBUGHISTOGRAM:
    import matplotlib.pyplot as plt


def local_sparsity_hybrid_wrapper(X, freq_width_energy=11, freq_width_sparsity=21, time_width=11, zeta = 80, double energy_criterium_db=-50):
    return local_sparsity_hybrid(X, freq_width_energy, freq_width_sparsity, time_width, zeta, energy_criterium_db)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef local_sparsity_hybrid(double[:,:,::1] X_orig, Py_ssize_t freq_width_energy, Py_ssize_t freq_width_sparsity, Py_ssize_t time_width, double zeta, double energy_criterium_db):

    cdef:
        Py_ssize_t P = X_orig.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X_orig.shape[1] # Eixo das frequências
        Py_ssize_t M = X_orig.shape[2] # Eixo do tempo
        
        Py_ssize_t freq_width_energy_lobe = (freq_width_energy-1)//2
        Py_ssize_t freq_width_sparsity_lobe = (freq_width_sparsity-1)//2
        Py_ssize_t max_freq_width_lobe
        Py_ssize_t time_width_lobe = (time_width-1)//2
        Py_ssize_t p, m, k, i, j

        double epsilon = 1e-10
        Py_ssize_t combined_size_sparsity = time_width * freq_width_sparsity

    IF DEBUGTIMER:
        cdef:
            clock_t time_i, time_f 
            double timer_initial_copy = 0, timer_initial_sort = 0, timer_new_merge = 0 


    max_freq_width_lobe = freq_width_energy_lobe
    if freq_width_sparsity_lobe > max_freq_width_lobe:
        max_freq_width_lobe = freq_width_sparsity_lobe  
    
    X_orig_ndarray = np.asarray(X_orig)
    # Realiza zero-padding no tensor de espectrogramas.
    X_ndarray = np.pad(X_orig, ((0, 0), (max_freq_width_lobe, max_freq_width_lobe), (time_width_lobe, time_width_lobe)))
    cdef double[:, :, :] X = X_ndarray

    # Calcula as janelas de Hamming utilizadas no algoritmo, separadamente para cada eixo.
    hamming_freq_energy_ndarray = np.hamming(freq_width_energy)
    hamming_freq_sparsity_ndarray = np.hamming(freq_width_sparsity)
    hamming_time_ndarray = np.hamming(time_width)
    hamming_asym_time_ndarray = np.hamming(time_width)
    hamming_asym_time_ndarray[time_width_lobe+1:] = 0

    hamming_energy = np.outer(hamming_freq_energy_ndarray, hamming_asym_time_ndarray)
    
    cdef double[:] hamming_freq_sparsity = hamming_freq_sparsity_ndarray
    cdef double[:] hamming_time = hamming_time_ndarray
    
    
    # Container que armazena uma janela de análise em um vetor.
    calc_vector_ndarray = np.zeros(combined_size_sparsity, dtype = np.double)
    cdef double[:] calc_vector = calc_vector_ndarray 

    # Container que armazena o resultado
    result_ndarray = np.zeros((K, M), dtype=np.double)
    cdef double[:, :] result = result_ndarray

    # Variáveis referentes ao cálculo da função de esparsidade
    sparsity_ndarray = epsilon * np.ones(P, dtype=np.double)
    cdef double[:] sparsity = sparsity_ndarray
    cdef double arr_norm, gini

    energy_ndarray = epsilon * np.ones((P, K, M), dtype=np.double)
    cdef double[:,:,:] energy = energy_ndarray


    # Variáveis referentes à combinação dos espectrogramas.
    cdef double[:] log_sparsity
    cdef double sum_log_sparsity
    combination_weight_ndarray = np.empty(P, dtype=np.double)
    cdef double[:] combination_weight = combination_weight_ndarray
    cdef double min_local_energy
    cdef double weights_sum

    # Variável utilizada para o critério híbrido.
    cdef double max_local_energy_db

    ############ Cálculo da função de energia local {{{ 

    IF DEBUGPRINT:
        print("Hamming window energy:")
        print_arr(np.outer(hamming_freq_energy_ndarray, hamming_asym_time_ndarray))

    IF DEBUGTIMER:
        time_i = clock()

    for p in range(P):
        energy_ndarray[p] = correlate(X_orig_ndarray[p], hamming_energy, mode='same')

    energy = energy_ndarray

    # ############ }}}

    IF DEBUGCOUNT:
        over_criterium = np.sum(10*np.log10(np.max(energy_ndarray, axis=0)) >= energy_criterium_db)
        proportion = over_criterium/(K * M)
        print(over_criterium, proportion)


    IF DEBUGHISTOGRAM:
        plt.figure()
        plt.hist(10*np.log10(energy_ndarray[0].flatten()), bins=50)

        plt.figure()
        plt.hist(10*np.log10(energy_ndarray[1].flatten()), bins=50)

        plt.figure()
        plt.hist(10*np.log10(energy_ndarray[2].flatten()), bins=50)

        plt.show()

        input(".")

    #cdef Py_ssize_t count1 = 0, count2 = 0

    ############ Combinação híbrida {{{

    # Itera pelos espectrogramas.
    
    for k in range(max_freq_width_lobe, K + max_freq_width_lobe):
        for m in range(time_width_lobe, M + time_width_lobe):
            red_k, red_m = k - max_freq_width_lobe, m - time_width_lobe # Apenas por simplicidade.

            # Encontra a maior energia local.
            max_local_energy_db = -INFINITY
            for p in range(P):
                if 10*log10(energy[p, red_k, red_m]) > max_local_energy_db:
                    max_local_energy_db = 10*log10(energy[p, red_k, red_m])
            
            # Se essa energia está abaixo do critério escolhido, realiza combinação por minmax. 
            if max_local_energy_db < energy_criterium_db:
                result[red_k, red_m] = INFINITY
                for p in range(P):
                    if X[p, k, m] < result[red_k, red_m]:
                        result[red_k, red_m] = X[p, k, m]

            # Caso contrário, realiza combinação por LS/SLS (no momento, só SLS implementado)
            else:
                for p in range(P):
                    # Copia a região janelada para o vetor de cálculo, multiplicando pelas janelas de Hamming.
                    for i in range(freq_width_sparsity):
                        for j in range(time_width):
                            calc_vector[i*time_width + j] = X[p, k - freq_width_sparsity_lobe + i, m - time_width_lobe + j] * \
                                    hamming_freq_sparsity[i] * hamming_time[j]        

                    IF DEBUGPRINT:
                        print("Vetor de cálculo:")
                        print(list(calc_vector))

                    # Ordena a região        
                    calc_vector_ndarray.sort()

                    IF DEBUGPRINT:
                        print("Vetor de cálculo ordenado:")
                        print(list(calc_vector))

                    # Calcula a função de esparsidade.
                    arr_norm = 0.0
                    gini = 0.0
                    
                    for i in range(combined_size_sparsity):
                        arr_norm = arr_norm + calc_vector[i]
                        gini = gini - 2*calc_vector[i] * (combined_size_sparsity - i - 0.5)/ (<double> combined_size_sparsity)

                    gini = 1 + gini/(arr_norm + epsilon)

                    # Índice para a matriz de esparsidade local deve ser ajustado porque essa não tem zero-padding.
                    sparsity[p] = epsilon + gini

                # Combinação (smoothed local sparsity apenas implementado):

                log_sparsity_ndarray = np.log(sparsity_ndarray)
                sum_log_sparsity = np.sum(log_sparsity_ndarray)

                log_sparsity = log_sparsity_ndarray

                min_local_energy = INFINITY
                weights_sum = 0.0
                for p in range(P):
                    combination_weight[p] = exp( (2*log_sparsity[p] - sum_log_sparsity) * zeta)
                    weights_sum += combination_weight[p]
                    if energy[p, red_k, red_m] < min_local_energy:
                        min_local_energy = energy[p, red_k, red_m]

                result[red_k, red_m] = 0.0
                for p in range(P):
                    result[red_k, red_m] = result[red_k, red_m] + X_orig[p, red_k, red_m] * combination_weight[p] * min_local_energy / energy[p, red_k, red_m]
                result[red_k, red_m] = result[red_k, red_m] / weights_sum

    # ############ }}}

    IF DEBUGPRINT:
        print("Energia")
        for p in range(P):
            print(f"Energia. p = {p}")
            print_arr(energy_ndarray[p])


    return result_ndarray















                
