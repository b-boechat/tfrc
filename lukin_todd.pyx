# Segunda versão otimizada, usando o merge with exclusion na horizontal também.


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

def lukin_todd_wrapper(X, freq_width=39, time_width=11, eta=8.0):
    return lukin_todd(X, freq_width, time_width, eta)


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef lukin_todd(double[:,:,::1] X_orig, Py_ssize_t freq_width, Py_ssize_t time_width, double eta):

    cdef:
        Py_ssize_t P = X_orig.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X_orig.shape[1] # Eixo das frequências
        Py_ssize_t M = X_orig.shape[2] # Eixo do tempo
        
        Py_ssize_t freq_width_lobe = (freq_width-1)//2
        Py_ssize_t time_width_lobe = (time_width-1)//2
        Py_ssize_t p, m, k, i_sort, j_sort, i, j
        double key

        double epsilon = 1e-10

    IF DEBUGTIMER:
        cdef:
            clock_t time_i, time_f 
            double timer_initial_copy = 0, timer_initial_sort = 0, timer_new_merge = 0 


    X_ndarray = np.pad(X_orig, ((0, 0), (freq_width_lobe, freq_width_lobe), (time_width_lobe, time_width_lobe)))
    cdef double[:, :, :] X = X_ndarray


    # Container que armazena um segmento horizontal de um espectrograma (com todos os bins de frequência). Usado nos cálculos.
    calc_region_ndarray = np.zeros((K + 2*freq_width_lobe, time_width), dtype = np.double)
    cdef double[:, :] calc_region = calc_region_ndarray 

    
    result_ndarray = np.zeros((K, M), dtype=np.double)
    cdef double[:, :] result = result_ndarray

    # Variáveis referentes ao merge inicial.
    cdef:
        Py_ssize_t num_vectors = freq_width
        Py_ssize_t len_vectors = time_width
        Py_ssize_t combined_size = freq_width*time_width
        Py_ssize_t j_parent, j_left_child, j_right_child, j_smaller_child, o
        Py_ssize_t element_origin, origin_index

    # Variáveis referentes ao merge com exclusão no eixo horizontal.
    cdef:
        double inclusion_scalar, exclusion_scalar, merge_buffer

        Py_ssize_t k_exc_orig, m_exc_orig # TODO temporário enquanto o programa não é restruturado em termos do sorted_region.


    # Heap que armazena o menor elemento não "consumido" de cada vetor.
    heap_elements_ndarray = np.empty(num_vectors, dtype=np.double)
    cdef double[:] heap_elements = heap_elements_ndarray 
    # Armazena o vetor de origem do elemento correspondente da heap.
    heap_origins_ndarray = np.empty(num_vectors, dtype=np.intp)
    cdef Py_ssize_t[:] heap_origins = heap_origins_ndarray 
    # Armazena o índice atual na heap de cada vetor.
    array_indices_ndarray = np.empty(num_vectors, dtype=np.intp)
    cdef Py_ssize_t[:] array_indices = array_indices_ndarray 
    # Armazena o vetor combinado de todos os elementos (nas iterações pares), usado para calcular a função de smearing.
    combined_even_ndarray = np.empty(combined_size, dtype=np.double)  
    cdef double[:] combined_even = combined_even_ndarray
    # Armazena o vetor combinado de todos os elementos (nas iterações ímpares), usado para calcular a função de smearing.
    combined_odd_ndarray = np.empty(combined_size, dtype=np.double)  
    cdef double[:] combined_odd = combined_odd_ndarray 

    # Memviews que apontam para combined_even e combined_odd, alternando em cada iteração.
    cdef double[:] combined
    cdef double[:] previous_combined

    # Variáveis referentes ao merge com exclusão.
    cdef:
        Py_ssize_t combined_index, previous_comb_index
        Py_ssize_t inclusion_index, exclusion_index
        Py_ssize_t inclusion_freq_position, exclusion_freq_position
        bint included_flag

    # Variáveis referentes ao cálculo da função de smearing.
    smearing_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] smearing = smearing_ndarray
    cdef double smearing_numerator, smearing_denominator

    # Variáveis referentes à combinação dos espectrogramas.
    cdef double weight, weights_sum, result_acc

    ############ Cálculo da função de smearing {{{

    # Itera pelos espectrogramas.
    for p in range(P):
        IF DEBUGPRINT:
            print(f"Padded X[{p}]")
            print_arr(X_ndarray[p], [freq_width_lobe, K + freq_width_lobe, time_width_lobe, M + time_width_lobe], colorama.Fore.CYAN)
        
        IF DEBUGTIMER:
            time_i = clock()

        # Copia a região inicial de cálculo para o container calc_region.
        for k in range(freq_width_lobe, K + freq_width_lobe): # Os limites não precisam incluir a região de zero-padding.
            for i in range(time_width):
                calc_region[k, i] = X[p, k, i]

        IF DEBUGTIMER:
            time_f = clock()
            timer_initial_copy = timer_initial_copy + <double> (time_f - time_i)

        IF DEBUGPRINT:
            print("Initial region:")
            print_arr(calc_region)

        # Itera pelos segmentos temporais.
        for m in range(time_width_lobe, M + time_width_lobe):

            if m == time_width_lobe:
                ##### Orderna os vetores horiontais do zero (só é feito uma vez por espectrograma, para os vetores do segmento mais à esquerda.) {{

                # Itera pelos bins de frequência
                for k in range(freq_width_lobe, K + freq_width_lobe): # Novamente, não precisa iterar pela região de zero-padding.

                    # Ordena o vetor horizontal.
                    
                    #calc_region_ndarray.sort()
                    for i_sort in range(1, time_width):
                        key = calc_region[k, i_sort]
                        j_sort = i_sort - 1
                        while j_sort >= 0 and key < calc_region[k, j_sort]:
                            calc_region[k, j_sort + 1] = calc_region[k, j_sort]
                            j_sort = j_sort - 1
                        calc_region[k, j_sort + 1] = key

                ##### }}

                IF DEBUGPRINT:
                    print("Initial sort:")
                    print_arr(calc_region)

            else:

                #### Obtém os próximos vetores horizontais, adicionando o elemento em m + time_width_lobe e 
                # excluindo o originalmente em m - time_width_lobe - 1 {{

                for k in range(freq_width_lobe, K + freq_width_lobe):
                    # Copia o valor a ser incluído e o valor a ser excluído para as variáveis correspondentes.
                    inclusion_scalar = X[p, k, m + time_width_lobe] #
                    exclusion_scalar = X[p, k, m - time_width_lobe - 1]

                    # Inicializa o índice.                    
                    i_sort = time_width - 1

                    # Realiza o algoritmo (TODO explicar)
                    while calc_region[k, i_sort] != exclusion_scalar:
                        if inclusion_scalar > calc_region[k, i_sort]:
                            calc_region[k, i_sort], inclusion_scalar = inclusion_scalar, calc_region[k, i_sort]

                        i_sort = i_sort - 1
                    
                    i_sort = i_sort - 1
                    while inclusion_scalar < calc_region[k, i_sort] and i_sort >= 0:
                        calc_region[k, i_sort + 1] = calc_region[k, i_sort]
                        i_sort = i_sort - 1

                    calc_region[k, i_sort + 1] = inclusion_scalar 


                ##### }}

                
            ##### Realiza o primeiro merge. {{
            combined = combined_odd
            previous_combined = combined_even
                
            for i in range(num_vectors):
                ### Inicializa a heap com o primeiro elemento de cada vetor {
                heap_elements[i] = calc_region[i, 0]
                heap_origins[i] = i
                array_indices[i] = 0
                ### }

                ### Heapify up. {
                j = i
                j_parent = (j - 1) // 2
                while j_parent >= 0 and heap_elements[j_parent] > heap_elements[j]:
                    heap_elements[j_parent], heap_elements[j] = heap_elements[j], heap_elements[j_parent]
                    heap_origins[j_parent], heap_origins[j] = i, heap_origins[j_parent]

                    j = j_parent
                    j_parent = (j - 1) // 2
                ### }
            for o in range(combined_size):
                ### Pop no primeiro elemento da heap {
                combined[o] = heap_elements[0]
                element_origin = heap_origins[0]
                array_indices[element_origin] += 1
                origin_index = array_indices[element_origin]
                if origin_index >= len_vectors:
                    heap_elements[0] = INFINITY
                else:
                    heap_elements[0] = calc_region[element_origin, origin_index]
                ### }

                ### Heapify down {           
                j = 0
                j_left_child = 2*j + 1
                while j_left_child < num_vectors:
                    j_smaller_child = j_left_child
                    j_right_child = j_left_child + 1
                    if j_right_child < num_vectors and heap_elements[j_right_child] < heap_elements[j_left_child]:
                        j_smaller_child = j_right_child

                    if heap_elements[j] <= heap_elements[j_smaller_child]:
                        break
                    
                    heap_elements[j], heap_elements[j_smaller_child] = heap_elements[j_smaller_child], heap_elements[j]
                    heap_origins[j], heap_origins[j_smaller_child] = heap_origins[j_smaller_child], heap_origins[j]

                    j = j_smaller_child
                    j_left_child = 2*j + 1

                ### }

            ##### }}

            ### Cálculo da função de smearing do Lukin-Todd {

            smearing_denominator = 0.0
            smearing_numerator = 0.0
            for o in range(combined_size):
                smearing_denominator = smearing_denominator + combined[o]
                smearing_numerator = smearing_numerator + (combined_size - o)*combined[o]
            smearing[p, 0, m - time_width_lobe] = smearing_numerator/(sqrt(smearing_denominator) + epsilon)

            ### }

            IF DEBUGPRINT:
                print("Window:")
                print_arr(X[p], [0, freq_width, m - time_width_lobe, m + time_width_lobe + 1], colorama.Fore.BLUE)
                print("Calc_region:")
                print_arr(calc_region, [0, freq_width, 0, time_width], colorama.Fore.MAGENTA)
                print("Combined vector:", list(combined))
                print_arr(smearing[p], [0, 1, m - time_width_lobe, m - time_width_lobe + 1], colorama.Fore.RED)

            # Itera pelos slices de frequência, exceto o primeiro.
            for k in range(freq_width_lobe + 1, K + freq_width_lobe):

                ### { Merge with exclusion

                combined, previous_combined = previous_combined, combined

                previous_comb_index = 0
                combined_index = 0
                inclusion_index = 0
                exclusion_index = 0

                for o in range(combined_size + len_vectors):
                    if previous_comb_index >= combined_size:
                        # Se os elementos de previous_combined já se esgotaram, pop num elemento de inclusion.
                        combined[combined_index] = calc_region[k + freq_width_lobe, inclusion_index]
                        combined_index = combined_index + 1
                        inclusion_index = inclusion_index + 1
                    elif exclusion_index < len_vectors and previous_combined[previous_comb_index] == calc_region[k - freq_width_lobe - 1, exclusion_index]:
                        # Pula elemento do previous_combined que faz parte do vetor exclusion.
                        previous_comb_index = previous_comb_index + 1
                        exclusion_index = exclusion_index + 1
                    elif inclusion_index >= len_vectors or previous_combined[previous_comb_index] <= calc_region[k + freq_width_lobe, inclusion_index]:
                        # Se os elementos de inclusion já se esgotaram, ou se o elemento atual de previous_combined é menor que o de inclusion, 
                        # pop num elemento de previous_combined
                        combined[combined_index] = previous_combined[previous_comb_index]
                        combined_index = combined_index + 1
                        previous_comb_index = previous_comb_index + 1
                    else:
                        # Por último, se o elemento atual de inclusion é menor que o de previous_combined, pop num elemento de inclusion.
                        combined[combined_index] = calc_region[k + freq_width_lobe, inclusion_index]
                        combined_index = combined_index + 1
                        inclusion_index = inclusion_index + 1

                ### }


                ### Função de smearing {
                smearing_denominator = 0.0
                smearing_numerator = 0.0
                for o in range(combined_size):
                    smearing_denominator = smearing_denominator + combined[o]
                    smearing_numerator = smearing_numerator + (combined_size-o)*combined[o]
                smearing[p, k - freq_width_lobe, m - time_width_lobe] = smearing_numerator/(sqrt(smearing_denominator) + epsilon)
                
                ### }

                IF DEBUGPRINT:
                    print("Window:")
                    print_arr(X[p], [k - freq_width_lobe, k + freq_width_lobe + 1, m - time_width_lobe, m + time_width_lobe + 1], colorama.Fore.BLUE)
                    print("Calc_region:")
                    print_arr(calc_region, [k - freq_width_lobe, k + freq_width_lobe + 1, 0, time_width], colorama.Fore.MAGENTA)
                    print("Combined vector:", list(combined))
                    print_arr(smearing[p], [k - freq_width_lobe, k - freq_width_lobe + 1, m - time_width_lobe, m - time_width_lobe + 1], colorama.Fore.RED)

    
    ############ }}}

    ############ Combinação dos espectrogramas {{{

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















                
