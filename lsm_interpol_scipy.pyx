# Versão Local Sparsity com interpolações, sem otimizações.


import numpy as np
from scipy.interpolate import RegularGridInterpolator
cimport cython
from libc.math cimport INFINITY, pow
DEF DEBUGPRINT = 0
DEF DEBUGTIMER = 0

IF DEBUGPRINT:
    import colorama
    from debug import print_arr

IF DEBUGTIMER:
    from libc.time cimport clock_t, clock, CLOCKS_PER_SEC


def local_sparsity_interpolation_scipy_wrapper(
    X, freq_width_energy=15, freq_width_sparsity=39, time_width=11, zeta = 80, 
    interpolation_steps=np.array([[4, 1], [2, 2], [1, 4]], dtype=np.intp)
    ):
    #print(f"freq_width_sparsity = {freq_width_sparsity}\nfreq_width_energy = {freq_width_energy}\ntime_width = {time_width}\nzeta = {zeta}")
    return local_sparsity_interpolation_scipy(X, freq_width_energy, freq_width_sparsity, time_width, zeta, interpolation_steps)

#@cython.boundscheck(False)
#@cython.wraparound(False) 
#@cython.nonecheck(False)
#@cython.cdivision(True)
cdef local_sparsity_interpolation_scipy(double[:,:,::1] X_orig, Py_ssize_t freq_width_energy, Py_ssize_t freq_width_sparsity, Py_ssize_t time_width, double zeta, Py_ssize_t[:,:] interpolation_steps):
    # Interpolation steps: P x 2

    cdef:
        Py_ssize_t P = X_orig.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X_orig.shape[1] # Eixo das frequências
        Py_ssize_t M = X_orig.shape[2] # Eixo do tempo
        
        Py_ssize_t freq_width_energy_lobe = (freq_width_energy-1)//2
        Py_ssize_t freq_width_sparsity_lobe = (freq_width_sparsity-1)//2
        Py_ssize_t max_freq_width_lobe
        Py_ssize_t time_width_lobe = (time_width-1)//2
        Py_ssize_t p, m, k, i, j
        Py_ssize_t max_multiple_k, max_multiple_m

        double epsilon = 1e-10
        Py_ssize_t combined_size_sparsity = time_width * freq_width_sparsity

    IF DEBUGTIMER:
        cdef:
            clock_t time_i, time_f 
            double timer_copy = 0.0
            double timer_sort = 0.0


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

    # Variáveis referentes à combinação dos espectrogramas.
    cdef double max_sparsity, min_local_energy, choosen_tfr_local_energy, sparsity_product, sparsity_ratio, sparsity_ratio_sum

    for p in range(P):
    
        IF DEBUGPRINT:
            print(f"Padded X[{p}]")
            print_arr(X_ndarray[p], [max_freq_width_lobe, K + max_freq_width_lobe, time_width_lobe, M + time_width_lobe], colorama.Fore.CYAN)
        
        ############ Cálculo da função de esparsidade local {{{

        # Itera pelas janelas de cálculo, levando em conta os passos de interpolação.
        for k in range(max_freq_width_lobe, K + max_freq_width_lobe, interpolation_steps[p, 0]):
            for m in range(time_width_lobe, M + time_width_lobe, interpolation_steps[p, 1]):

                IF DEBUGTIMER:
                    time_i = clock()

                # Copia a região janelada para o vetor de cálculo, multiplicando pelas janelas de Hamming.
                for i in range(freq_width_sparsity):
                    for j in range(time_width):
                        calc_vector[i*time_width + j] = X[p, k - freq_width_sparsity_lobe + i, m - time_width_lobe + j] * \
                                hamming_freq_sparsity[i] * hamming_time[j]        

                IF DEBUGTIMER:
                    time_f = clock()
                    timer_copy += time_f - time_i

                IF DEBUGPRINT:
                    print("Vetor de cálculo:")
                    print(list(calc_vector))

                IF DEBUGTIMER:
                    time_i = clock()
                
                # Ordena a região
                calc_vector_ndarray.sort()

                IF DEBUGTIMER:
                    time_f = clock()
                    timer_sort += time_f - time_i

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

        max_multiple_k = (K - 1)//interpolation_steps[p, 0] * interpolation_steps[p, 0]
        max_multiple_m = (M - 1)//interpolation_steps[p, 1] * interpolation_steps[p, 1]

        IF DEBUGPRINT:
            print(f"p = {p}\n")
            print("Região R1:")
            print_arr(X_ndarray[p], 
                [(max_freq_width_lobe + max_multiple_k) + 1, K + max_freq_width_lobe, 
                time_width_lobe, M + time_width_lobe], colorama.Fore.MAGENTA)


            print("Região R2:")
            print_arr(X_ndarray[p], 
                [max_freq_width_lobe, K + max_freq_width_lobe, 
                (time_width_lobe + max_multiple_m) + 1, M + time_width_lobe], colorama.Fore.MAGENTA)

        # Região R1

        for k in range( (max_freq_width_lobe + max_multiple_k) + 1, K + max_freq_width_lobe):
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
        for m in range( (time_width_lobe + max_multiple_m) + 1, M + time_width_lobe ):
            for k in range(max_freq_width_lobe, (max_freq_width_lobe + max_multiple_k) + 1):
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


        ############## }}}

        ############## {{{ Interpolação da esparsidade

        IF DEBUGPRINT:
            sparsity_orig = np.copy(sparsity_ndarray[p])

        k_exact_vals = np.arange(0, max_multiple_k, interpolation_steps[p, 0])
        m_exact_vals = np.arange(0, max_multiple_m, interpolation_steps[p, 1])

        kg, mg = np.meshgrid(k_exact_vals, m_exact_vals, indexing='ij', sparse=True)
        data = sparsity_ndarray[p][kg, mg]

        fit = RegularGridInterpolator((k_exact_vals, m_exact_vals), data)

        for k in range(1, max_multiple_k):
            for m in range(1, max_multiple_m):
                if k % interpolation_steps[p, 0] or m % interpolation_steps[p, 1]:
                    sparsity[p, k, m] = fit([k, m])[0]

        ############## }}} Interpolação da esparsidade

        IF DEBUGPRINT:
            print("Esparsidade")
            print("Original:")
            print_arr(sparsity_orig)
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


    IF DEBUGTIMER:
        print(f"Time copy: {timer_copy/CLOCKS_PER_SEC}\nTime sort: {timer_sort/CLOCKS_PER_SEC}")

    return result_ndarray















                
