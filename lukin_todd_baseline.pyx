# Versão baseline do Lukin-Todd, sem otimizações.


import numpy as np
cimport cython
from libc.math cimport INFINITY, sqrt, pow
DEF DEBUGPRINT = 0
DEF DEBUGTIMER = 0

IF DEBUGPRINT:
    import colorama
    from debug import print_arr


IF DEBUGTIMER:
    from libc.time cimport clock_t, clock

def lukin_todd_baseline_wrapper(X, freq_width=39, time_width=11, eta=8.0):
    return lukin_todd_baseline(X, freq_width, time_width, eta)


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef lukin_todd_baseline(double[:,:,::1] X_orig, Py_ssize_t freq_width, Py_ssize_t time_width, double eta):

    cdef:
        Py_ssize_t P = X_orig.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X_orig.shape[1] # Eixo das frequências
        Py_ssize_t M = X_orig.shape[2] # Eixo do tempo
        
        Py_ssize_t freq_width_lobe = (freq_width-1)//2
        Py_ssize_t time_width_lobe = (time_width-1)//2
        Py_ssize_t p, m, k, i, j

        #Py_ssize_t i_sort, j_sort
        #double key

        double epsilon = 1e-10
        Py_ssize_t combined_size = time_width * freq_width

    IF DEBUGTIMER:
        cdef:
            clock_t time_i, time_f 
            double timer_initial_copy = 0, timer_initial_sort = 0, timer_new_merge = 0 


    # Realiza zero-padding no tensor de espectrogramas.
    X_ndarray = np.pad(X_orig, ((0, 0), (freq_width_lobe, freq_width_lobe), (time_width_lobe, time_width_lobe)))
    cdef double[:, :, :] X = X_ndarray


    # Container que armazena uma janela de análise em um vetor.
    calc_vector_ndarray = np.zeros(combined_size, dtype = np.double)
    cdef double[:] calc_vector = calc_vector_ndarray 
 
    # Container que armazena o resultado
    result_ndarray = np.zeros((K, M), dtype=np.double)
    cdef double[:, :] result = result_ndarray

    # Variáveis referentes ao cálculo da função de smearing.
    smearing_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] smearing = smearing_ndarray
    cdef double smearing_numerator, smearing_denominator
    cdef Py_ssize_t o

    # Variáveis referentes à combinação dos espectrogramas.
    cdef double weight, weights_sum, result_acc

    ############ Cálculo da função de smearing {{{

    # Itera pelos espectrogramas.
    for p in range(P):
        IF DEBUGPRINT:
            print(f"Padded X[{p}]")
            print_arr(X_ndarray[p], [freq_width_lobe, K + freq_width_lobe, time_width_lobe, M + time_width_lobe], colorama.Fore.CYAN)
        # Itera pelos segmentos temporais.
        
        IF DEBUGTIMER:
            time_i = clock()

        for k in range(freq_width_lobe, K + freq_width_lobe):
            for m in range(time_width_lobe, M + time_width_lobe):

                # Copia a região janelada para o vetor de cálculo. 
                for i in range(freq_width):
                    for j in range(time_width):
                        calc_vector[i*time_width + j] = X[p, k - freq_width_lobe + i, m - time_width_lobe + j]                

                IF DEBUGPRINT:
                    print("Vetor de cálculo:")
                    print(list(calc_vector))

                # Ordena a região
                
                calc_vector_ndarray.sort()

                #for i_sort in range(1, combined_size):
                #    key = calc_vector[i_sort]
                #    j_sort = i_sort - 1
                #    while j_sort >= 0 and key < calc_vector[j_sort]:
                #        calc_vector[j_sort + 1] = calc_vector[j_sort]
                #        j_sort = j_sort - 1
                #    calc_vector[j_sort + 1] = key        


                IF DEBUGPRINT:
                    print("Vetor de cálculo ordenado:")
                    print(list(calc_vector))

                # Calcula a função de smearing.
                smearing_denominator = 0.0
                smearing_numerator = 0.0
                for o in range(combined_size):
                    smearing_denominator = smearing_denominator + calc_vector[o]
                    smearing_numerator = smearing_numerator + (combined_size-o)*calc_vector[o]
                smearing[p, k - freq_width_lobe, m - time_width_lobe] = smearing_numerator/(sqrt(smearing_denominator) + epsilon)
                

    
    ############ }}}

    ############ Combinação dos espectrogramas {{{

    IF DEBUGPRINT:
        for p in range(P):
            print(f"Smearing ({p})")
            print_arr(smearing[p])

    # TODO tratar o caso não smoothed.

    for k in range(K):
        for m in range(M):
            weights_sum = 0.0
            result_acc = 0.0
            for p in range(P):
                weight = 1./(pow(smearing[p, k, m], eta) + epsilon)
                result_acc = result_acc + weight * X_orig[p, k, m]
                weights_sum = weights_sum + weight
            result[k, m] = result_acc / weights_sum

    ############ }}}

    return result_ndarray















                
