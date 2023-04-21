# Versão baseline do Local Sparsity, sem otimizações.


import numpy as np
cimport cython
from libc.math cimport INFINITY, pow
DEF DEBUGPRINT = 0
DEF DEBUGTIMER = 1

IF DEBUGPRINT:
    import colorama
    from debug import print_arr

IF DEBUGTIMER:
    from libc.time cimport clock_t, clock, CLOCKS_PER_SEC


def local_sparsity_baseline_wrapper(X, freq_width_energy=15, freq_width_sparsity=39, time_width=11, zeta = 80):
    #print(f"freq_width_sparsity = {freq_width_sparsity}\nfreq_width_energy = {freq_width_energy}\ntime_width = {time_width}\nzeta = {zeta}")
    return local_sparsity_baseline(X, freq_width_energy, freq_width_sparsity, time_width, zeta)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef local_sparsity_baseline(double[:,:,::1] X_orig, Py_ssize_t freq_width_energy, Py_ssize_t freq_width_sparsity, Py_ssize_t time_width, double zeta):

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
            double timer_sparsity, timer_energy


    max_freq_width_lobe = freq_width_energy_lobe
    if freq_width_sparsity_lobe > max_freq_width_lobe:
        max_freq_width_lobe = freq_width_sparsity_lobe  
    
    # Realiza zero-padding no tensor de espectrogramas.
    X_ndarray = np.pad(X_orig, ((0, 0), (max_freq_width_lobe, max_freq_width_lobe), (time_width_lobe, time_width_lobe)))
    cdef double[:, :, :] X = X_ndarray

    # Calcula as janelas de Hamming utilizadas no algoritmo, separadamente para cada eixo.
    hamming_freq_energy_ndarray = np.hamming(freq_width_energy)
    hamming_freq_sparsity_ndarray = np.hamming(freq_width_sparsity)
    hamming_time_ndarray = np.hamming(time_width)
    hamming_asym_time_ndarray = np.hamming(time_width)
    hamming_asym_time_ndarray[time_width_lobe+1:] = 0

    # hamming_freq_sparsity_ndarray = 2*np.ones(freq_width_sparsity)
    # hamming_time_ndarray = 3*np.ones(time_width)
    # hamming_freq_energy_ndarray = 2*np.ones(freq_width_energy)
    # hamming_asym_time_ndarray = 5*np.ones(time_width)
    
    cdef double[:] hamming_freq_energy = hamming_freq_energy_ndarray
    cdef double[:] hamming_freq_sparsity = hamming_freq_sparsity_ndarray
    cdef double[:] hamming_asym_time = hamming_asym_time_ndarray
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
    cdef double max_sparsity, min_local_energy, choosen_tfr_local_energy, sparsity_product, sparsity_ratio, sparsity_ratio_sum

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

                #for i_sort in range(1, combined_size_sparsity):
                #    key = calc_vector[i_sort]
                #    j_sort = i_sort - 1
                #    while j_sort >= 0 and key < calc_vector[j_sort]:
                #        calc_vector[j_sort + 1] = calc_vector[j_sort]
                #        j_sort = j_sort - 1
                #    calc_vector[j_sort + 1] = key        

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


    ############ Cálculo da função de energia local {{{ 

    IF DEBUGTIMER:
        time_i = clock()

    for p in range(P):
        for k in range(max_freq_width_lobe, K + max_freq_width_lobe):
           for m in range(time_width_lobe, M + time_width_lobe):
                # Calcula a energia local no segmento definido por (p, k, m).
                
                for i in range(freq_width_energy):
                   for j in range(time_width):
                       energy[p, k - max_freq_width_lobe, m - time_width_lobe] = energy[p, k - max_freq_width_lobe, m - time_width_lobe] + X[p, k - freq_width_energy_lobe + i, m - time_width_lobe + j] \
                           * hamming_freq_energy[i] * hamming_asym_time[j]

                IF DEBUGPRINT:
                    print_arr(X_ndarray[p], [k - freq_width_energy_lobe, k + freq_width_energy_lobe + 1, m - time_width_lobe, m + time_width_lobe + 1])
                    print(f"Energy (p = {p}, k = {k} - {max_freq_width_lobe}, m = {m} - {time_width_lobe})")
                    print_arr(energy_ndarray[p], [k - max_freq_width_lobe, k - max_freq_width_lobe + 1, m - time_width_lobe, m - time_width_lobe + 1])



    # ############ }}}

    IF DEBUGTIMER:
        time_f = clock()
        timer_energy = (<double> (time_f - time_i) ) / CLOCKS_PER_SEC 
        print(f"Timer sparsity: {timer_sparsity}\nTimer energy: {timer_energy}")

    # ############ Combinação por Esparsidade Local e compensação por Energia Local {{
        
    if zeta < 0: # Local Sparsity Method (not smoothed)
        for k in range(K): 
            for m in range(M):
                max_sparsity = -1.0
                min_local_energy = INFINITY
                for p in range(P):
                    if sparsity[p, k, m] > max_sparsity:
                        result[k, m] = X[p, k + max_freq_width_lobe, m + time_width_lobe]
                        max_sparsity = sparsity[p, k, m]
                        choosen_tfr_local_energy = energy[p, k, m]      
                    if energy[p, k, m] < min_local_energy:
                        min_local_energy = energy[p, k, m]
                result[k, m] *= min_local_energy/choosen_tfr_local_energy

    else: # Smoothed Local Sparsity Method

        # Itera pelos bins de frequência.
        for k in range(K): 
            # Itera pelos segmentos temporais.
            for m in range(M):
                # Calcula a menor energia local e o produto de esparsidades locais, iterando pelos espectrogramas.
                IF DEBUGPRINT:
                    print(f"\n\nk = {k}, m = {m}\n")

                min_local_energy = INFINITY
                sparsity_product = 1.0
                for p in range(P):
                    sparsity_product *= sparsity[p, k, m]
                    if energy[p, k, m] < min_local_energy:
                        min_local_energy = energy[p, k, m]
                IF DEBUGPRINT:
                    print(f"sparsity product = {sparsity_product:.2f}, min_local_energy = {min_local_energy:.2f}")

                # Itera pelos espectrogramas novamente, calculando a razão de esparsidade e computando o resultado incrementalmente.
                sparsity_ratio_sum = epsilon
                for p in range(P):
                    sparsity_ratio = pow(sparsity[p, k, m] * sparsity[p, k, m] / sparsity_product, zeta) # Dois fatores sparsity[p, k, m] para removê-lo do produto.

                    IF DEBUGPRINT:
                        print(f"sparsity ratio (p = {p}) = {sparsity_ratio:.2f}")

                    sparsity_ratio_sum += sparsity_ratio 
                    result[k, m] += X[p, k + max_freq_width_lobe, m + time_width_lobe] * sparsity_ratio *  min_local_energy / energy[p, k, m]

                # Divide o resultado pela soma das razões de esparsidade. 
                result[k, m] /= sparsity_ratio_sum 


    ############ }} Combinação por Esparsidade Local e compensação por Energia Local

    IF DEBUGPRINT:
        print("Energia")
        for p in range(P):
            print(f"Energia. p = {p}")
            print_arr(energy_ndarray[p])


    return result_ndarray















                
