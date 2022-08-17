# Versão baseline do Local Sparsity, sem otimizações.


import numpy as np
cimport cython
from libc.math cimport INFINITY, pow
DEF DEBUGPRINT = 0
DEF DEBUGTIMER = 0

IF DEBUGPRINT:
    import colorama
    from debug import print_arr

IF DEBUGTIMER:
    from libc.time cimport clock_t, clock


def local_sparsity_baseline_interpolation_wrapper(X, freq_width_energy=15, freq_width_sparsity=39, time_width=11, zeta = 80):
    #print(f"freq_width_sparsity = {freq_width_sparsity}\nfreq_width_energy = {freq_width_energy}\ntime_width = {time_width}\nzeta = {zeta}")
    return local_sparsity_baseline_interpolation(X, freq_width_energy, freq_width_sparsity, time_width, zeta)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef local_sparsity_baseline_interpolation(double[:,:,::1] X_orig, Py_ssize_t freq_width_energy, Py_ssize_t freq_width_sparsity, Py_ssize_t time_width, double zeta):

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
            double timer_initial_copy = 0, timer_initial_sort = 0, timer_new_merge = 0 


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

    # Container e variáveis referentes ao cálculo da função de esparsidade
    sparsity_ndarray = epsilon * np.ones((P, K, M), dtype=np.double)
    cdef double[:,:,:] sparsity = sparsity_ndarray
    cdef double arr_norm, gini

    # Container que armazena as energias locais. 
    energy_ndarray = epsilon * np.ones((P, K, M), dtype=np.double)
    cdef double[:,:,:] energy = energy_ndarray

    # Armazena o passo de interpolação em cada direção. interpolation_steps[i, j] -> Interpolações para p = i. j = 0: na frequência; j = 1: no tempo
    interpolation_steps_ndarray = np.array([[4, 1], [2, 2], [1, 4]], dtype=np.intp)
    cdef Py_ssize_t[:,:] interpolation_steps = interpolation_steps_ndarray

    # Variáveis referentes à combinação dos espectrogramas.
    cdef double max_sparsity, min_local_energy, choosen_tfr_local_energy, sparsity_product, sparsity_ratio, sparsity_ratio_sum

    ############ Cálculo da função de esparsidade local {{{

    for p in range(P):
    
        IF DEBUGPRINT:
            print(f"Padded X[{p}]")
            print_arr(X_ndarray[p], [max_freq_width_lobe, K + max_freq_width_lobe, time_width_lobe, M + time_width_lobe], colorama.Fore.CYAN)
        
        # Itera pelas janelas de cálculo, levando em conta os passos de interpolação.
        for k in range(max_freq_width_lobe, K + max_freq_width_lobe, interpolation_steps[p, 0]):
            for m in range(time_width_lobe, M + time_width_lobe, interpolation_steps[p, 1]):
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

        # Devido ao passo de interpolação na janela, é possível que faltem algumas janelas no fim.
        # Por exemplo, para a frequência: sendo mfwl := max_freq_width_lobe e isf := interpolation_steps[p, 0]
        # As janelas pré-interpolação estão nas posições de frequência: mfwl, mfwl + isf, mfwl + 2isf, ... , mfwl + (K - 1)//isf * isf
        # Analogamente, sendo twl := time_width_lobe e ist := interpolation_steps[p, 1],
        # A última posição no tempo das janelas é twl + (M - 1)//ist * ist
        # Logo, é preciso completar o cálculo na região R1 U R2, sendo R1 = {(k, m): k > (K - 1)//isf * isf} e R2 = {(k, m): m > twl + (M - 1)//ist * ist} 

        IF DEBUGPRINT:
            print(f"p = {p}\n")
            print("Região R1:")
            print_arr(X_ndarray[p], 
                [(max_freq_width_lobe + (K - 1)//interpolation_steps[p, 0] * interpolation_steps[p, 0]) + 1, K + max_freq_width_lobe, 
                time_width_lobe, M + time_width_lobe], colorama.Fore.MAGENTA)


            print("Região R2:")
            print_arr(X_ndarray[p], 
                [max_freq_width_lobe, K + max_freq_width_lobe, 
                (time_width_lobe + (M - 1)//interpolation_steps[p, 1] * interpolation_steps[p, 1]) + 1, M + time_width_lobe], colorama.Fore.MAGENTA)

        # Região R1

        for k in range( (max_freq_width_lobe + (K - 1)//interpolation_steps[p, 0] * interpolation_steps[p, 0]) + 1, K + max_freq_width_lobe):
            for m in range(time_width_lobe, M + time_width_lobe):
                for i in range(freq_width_sparsity):
                    for j in range(time_width):
                        calc_vector[i*time_width + j] = X[p, k - freq_width_sparsity_lobe + i, m - time_width_lobe + j] * \
                                hamming_freq_sparsity[i] * hamming_time[j] 
                calc_vector_ndarray.sort()
                arr_norm = 0.0
                gini = 0.0
                for i in range(combined_size_sparsity):
                    arr_norm = arr_norm + calc_vector[i]
                    gini = gini - 2*calc_vector[i] * (combined_size_sparsity - i - 0.5)/ (<double> combined_size_sparsity)
                gini = 1 + gini/(arr_norm + epsilon)
                sparsity[p, k - max_freq_width_lobe, m - time_width_lobe] += gini

        # Região R2 - R1
        for m in range( (time_width_lobe + (M - 1)//interpolation_steps[p, 1] * interpolation_steps[p, 1]) + 1, M + time_width_lobe ):
            for k in range(max_freq_width_lobe, (max_freq_width_lobe + (K - 1)//interpolation_steps[p, 0] * interpolation_steps[p, 0]) + 1):
                for i in range(freq_width_sparsity):
                    for j in range(time_width):
                        calc_vector[i*time_width + j] = X[p, k - freq_width_sparsity_lobe + i, m - time_width_lobe + j] * \
                                hamming_freq_sparsity[i] * hamming_time[j] 
                calc_vector_ndarray.sort()
                arr_norm = 0.0
                gini = 0.0
                for i in range(combined_size_sparsity):
                    arr_norm = arr_norm + calc_vector[i]
                    gini = gini - 2*calc_vector[i] * (combined_size_sparsity - i - 0.5)/ (<double> combined_size_sparsity)
                gini = 1 + gini/(arr_norm + epsilon)
                sparsity[p, k - max_freq_width_lobe, m - time_width_lobe] += gini


    # ############ }}}

    ############## {{{ Interpolação da esparsidade

    IF DEBUGPRINT:
        sparsity_orig = np.copy(sparsity_ndarray)

    cdef Py_ssize_t max_row_slice, max_col_slice

    # A interpopulação é feita diferentemente para cada resolução, suponto que estão dispostas da seguinte forma.
    # Índice 0: 1024 pontos: Interpolação de 4 em 4 na frequência.
    # Índice 1: 2048 pontos: Interpolação de 2 em 2 na frequência e 2 em 2 no tempo.
    # Índice 2: 4096 pontos: Interpolação de 4 em 4 no tempo.

    # Índice 0 (1024)

    max_row_slice = ((K - 1) // 4) * 4 + 1
    max_col_slice = M

    for i in range(1, max_row_slice, 4):
        for j in range(max_col_slice):  
            #print(f"({i}, {(j)}) Início: {sparsity[0, i - 1, j]:.3f}. Fim: {sparsity[0, i + 3, j]:.3f}. Step: {(sparsity[0, i + 3, j] - sparsity[0, i - 1, j])/4:.3f}")
            sparsity[0, i, j] = sparsity[0, i - 1, j] + (sparsity[0, i + 3, j] - sparsity[0, i - 1, j])/4
            sparsity[0, i + 1, j] = sparsity[0, i - 1, j] + (sparsity[0, i + 3, j] - sparsity[0, i - 1, j])*2/4
            sparsity[0, i + 2, j] = sparsity[0, i - 1, j] + (sparsity[0, i + 3, j] - sparsity[0, i - 1, j])*3/4

    # Índice 1 (2048)
    
    max_row_slice = ((K - 1) // 2) * 2 + 1
    max_col_slice = ((M - 1) // 2) * 2 + 1
    
    for i in range(1, max_row_slice, 2):
        for j in range(0, max_col_slice, 2): 
            sparsity[1, i, j] = (sparsity[1, i - 1, j] + sparsity[1, i + 1, j])/2

    for j in range(1, max_col_slice, 2):
        for i in range(max_row_slice): 
            sparsity[1, i, j] = (sparsity[1, i, j - 1] + sparsity[1, i, j + 1])/2

    # Índice 2 (4096)

    max_row_slice = K
    max_col_slice = ((M - 1) // 4) * 4 + 1

    for j in range(1, max_col_slice, 4):
        for i in range(max_row_slice):
            sparsity[2, i, j] = sparsity[2, i, j - 1] + (sparsity[2, i, j + 3] - sparsity[2, i, j - 1])/4
            sparsity[2, i, j + 1] = sparsity[2, i, j - 1] + (sparsity[2, i, j + 3] - sparsity[2, i, j - 1])*2/4
            sparsity[2, i, j + 2] = sparsity[2, i, j - 1] + (sparsity[2, i, j + 3] - sparsity[2, i, j - 1])*3/4


    ############## }}} Interpolação da esparsidade

    IF DEBUGPRINT:
        print("Esparsidade")
        for p in range(P):
            print(f"p = {p}")
            print("Original:")
            print_arr(sparsity_orig[p])
            print("Interpolado")
            print_arr(sparsity[p])


    ############ Cálculo da função de energia local {{{ 

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



    ############## }}}


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















                
