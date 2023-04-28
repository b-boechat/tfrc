# Versão do Local Sparsity com baseline no cálculo da esparsidade e otimizações nas demais etapas.


import numpy as np
from scipy.signal import correlate
cimport cython
from libc.math cimport INFINITY, exp
DEF DEBUGPRINT = 0
DEF DEBUGTIMER = 1

IF DEBUGPRINT:
    import colorama
    from debug import print_arr

IF DEBUGTIMER:
    from libc.time cimport clock_t, clock, CLOCKS_PER_SEC


def local_sparsity_baseline_opt_wrapper(X, freq_width_energy=15, freq_width_sparsity=39, time_width=11, zeta = 80):
    #print(f"freq_width_sparsity = {freq_width_sparsity}\nfreq_width_energy = {freq_width_energy}\ntime_width = {time_width}\nzeta = {zeta}")
    return local_sparsity_baseline_opt(X, freq_width_energy, freq_width_sparsity, time_width, zeta)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef local_sparsity_baseline_opt(double[:,:,::1] X_orig, Py_ssize_t freq_width_energy, Py_ssize_t freq_width_sparsity, Py_ssize_t time_width, double zeta):

    cdef:
        Py_ssize_t P = X_orig.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X_orig.shape[1] # Eixo das frequências
        Py_ssize_t M = X_orig.shape[2] # Eixo do tempo
        
        Py_ssize_t freq_width_energy_lobe = (freq_width_energy-1)//2
        Py_ssize_t freq_width_sparsity_lobe = (freq_width_sparsity-1)//2
        Py_ssize_t max_freq_width_lobe
        Py_ssize_t time_width_lobe = (time_width-1)//2
        Py_ssize_t p, m, k, i, j

        #Py_ssize_t i_sort, j_sort
        #double key

        double epsilon = 1e-10
        Py_ssize_t combined_size_sparsity = time_width * freq_width_sparsity

    IF DEBUGTIMER:
        cdef:
            clock_t time_i, time_f 
            double timer_sparsity, timer_energy, timer_comb


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
    sparsity_ndarray = epsilon * np.ones((P, K, M), dtype=np.double)
    cdef double[:,:,:] sparsity = sparsity_ndarray
    cdef double arr_norm, gini

    energy_ndarray = epsilon * np.ones((P, K, M), dtype=np.double)
    cdef double[:,:,:] energy = energy_ndarray

    # Variáveis referentes à combinação dos espectrogramas.
    cdef double[:,:] min_energy
    cdef Py_ssize_t[:,:] choosen_p

    cdef double[:, :, :] log_sparsity
    cdef double[:, :] sum_log_sparsity

    combination_weight_ndarray = np.empty((P, K, M), dtype=np.double)
    cdef double[:, :, :] combination_weight = combination_weight_ndarray


    ############ Cálculo da função de energia local {{{ 

    IF DEBUGPRINT:
        print("Hamming window energy:")
        print_arr(np.outer(hamming_freq_energy_ndarray, hamming_asym_time_ndarray))

    IF DEBUGTIMER:
        time_i = clock()

    for p in range(P):
        energy_ndarray[p] = correlate(X_orig_ndarray[p], hamming_energy, mode='same')

        IF DEBUGPRINT:
            print(f"Energy (p = {p})")
            print_arr(energy_ndarray[p])

    energy = energy_ndarray

    # ############ }}}

    IF DEBUGTIMER:
        time_f = clock()
        timer_energy = (<double> (time_f - time_i) ) / CLOCKS_PER_SEC 

    ############ Cálculo da função de esparsidade local {{{

    IF DEBUGTIMER:
        time_i = clock()

    # Itera pelos espectrogramas.
    for p in range(P):
        IF DEBUGPRINT:
            print(f"Padded X[{p}]")
            print_arr(X_ndarray[p], [max_freq_width_lobe, K + max_freq_width_lobe, time_width_lobe, M + time_width_lobe], colorama.Fore.CYAN)

        for k in range(max_freq_width_lobe, K + max_freq_width_lobe):
            for m in range(time_width_lobe, M + time_width_lobe):

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
                sparsity[p, k - max_freq_width_lobe, m - time_width_lobe] += gini

    
    # ############ }}}

    IF DEBUGTIMER:
        time_f = clock()
        timer_sparsity = (<double> (time_f - time_i) ) / CLOCKS_PER_SEC 

    IF DEBUGTIMER:
        time_i = clock()

    # ############ Combinação por Esparsidade Local e compensação por Energia Local {{
        
    if zeta < 0: # Local Sparsity Method (not smoothed)
        
        min_energy_ndarray = np.min(energy_ndarray, axis=0)
        min_energy = min_energy_ndarray

        choosen_p_ndarray = np.argmax(sparsity_ndarray, axis=0)
        choosen_p = choosen_p_ndarray

        for k in range(K): 
            for m in range(M):
                result[k, m] = X_orig[choosen_p[k, m], k, m] * min_energy[k, m] / energy[p, k, m]

    else: # Smoothed Local Sparsity Method

        log_sparsity_ndarray = np.log(sparsity_ndarray)
        sum_log_sparsity_ndarray = np.sum(log_sparsity_ndarray, axis=0)

        log_sparsity = log_sparsity_ndarray
        sum_log_sparsity = sum_log_sparsity_ndarray

        for p in range(P):
            for k in range(K): 
                for m in range(M):
                    combination_weight[p, k, m] = exp( (2*log_sparsity[p, k, m] - sum_log_sparsity[k, m]) * zeta)
        
        result_ndarray = np.average(X_orig_ndarray * np.min(energy_ndarray, axis=0)/energy_ndarray, axis=0, weights=combination_weight_ndarray)


    ############ }} Combinação por Esparsidade Local e compensação por Energia Local

    IF DEBUGTIMER:
        time_f = clock()
        timer_comb = (<double> (time_f - time_i) ) / CLOCKS_PER_SEC 
        print(f"Timer sparsity: {timer_sparsity}\nTimer energy: {timer_energy}\nTimer comb: {timer_comb}")

    IF DEBUGPRINT:
        print("Energia")
        for p in range(P):
            print(f"Energia. p = {p}")
            print_arr(energy_ndarray[p])


    return result_ndarray