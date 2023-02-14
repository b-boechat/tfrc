import numpy as np
DEF DEBUGTIMER = 0
DEF DEBUGPRINT = 0

cimport cython
from libc.math cimport INFINITY, pow 

IF DEBUGPRINT:
    import colorama
    from debug import print_arr

IF DEBUGTIMER:
    from libc.time cimport clock_t, clock, CLOCKS_PER_SEC

def local_sparsity_wrapper(X, freq_width_energy=15, freq_width_sparsity=39, time_width=11, zeta = 80):
    #print(f"freq_width_sparsity = {freq_width_sparsity}\nfreq_width_energy = {freq_width_energy}\ntime_width = {time_width}\nzeta = {zeta}")
    return local_sparsity(X, freq_width_energy, freq_width_sparsity, time_width, zeta)



@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef local_sparsity(double[:,:,::1] X_orig, Py_ssize_t freq_width_energy, Py_ssize_t freq_width_sparsity, Py_ssize_t time_width, double zeta):
    
    cdef:
        Py_ssize_t P = X_orig.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X_orig.shape[1] # Eixo das frequências
        Py_ssize_t M = X_orig.shape[2] # Eixo do tempo

        Py_ssize_t freq_width_energy_lobe = (freq_width_energy-1)//2
        Py_ssize_t freq_width_sparsity_lobe = (freq_width_sparsity-1)//2
        Py_ssize_t time_width_lobe = (time_width-1)//2
        Py_ssize_t p, k, aux_k, m, i, j, i_sort, j_sort

        double key, max_sparsity, sparsity_product, sparsity_ratio, sparsity_ratio_sum, min_local_energy, choosen_tfr_local_energy

        Py_ssize_t max_freq_width_lobe

        double epsilon = 1e-10


    IF DEBUGTIMER:
        cdef:
            clock_t time_i, time_f, time_ii, time_ff 
            double timer_sparsity = 0, timer_energy = 0, timer_sort = 0, timer_comb = 0
    
    sparsity_ndarray = epsilon * np.ones((P, K, M), dtype=np.double)
    energy_ndarray = epsilon * np.ones((P, K, M), dtype=np.double) # Matriz de energias é ser inicializada com epsilon (para evitar divisão por 0) e calculada incrementalmente.
    result_ndarray = np.zeros((K, M), dtype=np.double) # Matriz resultado precisa ser inicializada com 0 no Smoothed Local Sparsity.
    
    cdef double[:, :, :] sparsity = sparsity_ndarray
    cdef double[:, :, :] energy = energy_ndarray
    cdef double[:, :] result = result_ndarray

    max_freq_width_lobe = freq_width_energy_lobe
    if freq_width_sparsity_lobe > max_freq_width_lobe:
        max_freq_width_lobe = freq_width_sparsity_lobe  

    X_ndarray = np.pad(X_orig, ((0, 0), (max_freq_width_lobe, max_freq_width_lobe), (time_width_lobe, time_width_lobe)))
    cdef double[:, :, :] X = X_ndarray

    IF DEBUGPRINT:
        for p in range(P):
            print(f"p = {p}", end="\n\n")
            print_arr(X[p])

    # Calcula as janelas de Hamming utilizadas no algoritmo, separadamente para cada eixo.
    hamming_freq_energy_ndarray = np.hamming(freq_width_energy)
    hamming_freq_sparsity_ndarray = np.hamming(freq_width_sparsity)
    hamming_time_ndarray = np.hamming(time_width)
    hamming_asym_time_ndarray = np.hamming(time_width)
    hamming_asym_time_ndarray[time_width_lobe+1:] = 0

    # DEBUG
    # hamming_freq_sparsity_ndarray = 2*np.ones(freq_width_sparsity)
    # hamming_time_ndarray = 3*np.ones(time_width)
    # hamming_freq_energy_ndarray = 2*np.ones(freq_width_energy)
    # hamming_asym_time_ndarray = 5*np.ones(time_width)


    cdef double[:] hamming_freq_energy = hamming_freq_energy_ndarray
    cdef double[:] hamming_freq_sparsity = hamming_freq_sparsity_ndarray
    cdef double[:] hamming_asym_time = hamming_asym_time_ndarray
    cdef double[:] hamming_time = hamming_time_ndarray

    sort_indices_ndarray = np.empty((K, time_width), dtype=np.intp)
    cdef Py_ssize_t[:,:] sort_indices = sort_indices_ndarray

    aux_horiz_vector_ndarray = np.empty(time_width, dtype=np.double)
    cdef double[:] aux_horiz_vector = aux_horiz_vector_ndarray

    ######### {Variáveis para a combinação de vetores ordenados por heap e cálculo do índice de Gini
    cdef:
        Py_ssize_t num_vectors = freq_width_sparsity
        Py_ssize_t len_vectors = time_width
        Py_ssize_t combined_size = num_vectors * len_vectors
        Py_ssize_t j_parent, j_left_child, j_right_child, j_smaller_child, o
        Py_ssize_t element_origin, origin_index


    heap_elements_ndarray = np.empty(num_vectors, dtype=np.double)
    heap_origins_ndarray = np.empty(num_vectors, dtype=np.intp)
    array_indices_ndarray = np.empty(num_vectors, dtype=np.intp)
    combined_ndarray = np.empty(combined_size, dtype=np.double)

    cdef double[:] heap_elements = heap_elements_ndarray # Heap que armazena o menor elemento não "consumido" de cada vetor.
    cdef Py_ssize_t[:] heap_origins = heap_origins_ndarray # Armazena o vetor de origem do elemento correspondente da heap.
    cdef Py_ssize_t[:] array_indices = array_indices_ndarray # Armazena o índice atual na heap de cada vetor.
    cdef double[:] combined = combined_ndarray # Armazena o vetor combinado de todos os elementos

    cdef double combined_size_db, arr_norm, gini

    ######### }

    IF DEBUGTIMER:
        time_i = clock() 

    ################################################### Cálculo da Esparsidade Local {{ 

    # Itera pelos espectrogramas.
    for p in range(P):
        # Itera pelos segmentos temporais.
        for m in range(time_width_lobe, M + time_width_lobe):

            ################################### Ordenação dos vetores horizontais {

            IF DEBUGTIMER:
                time_ii = clock()

            # Itera pelos bins de frequência
            for k in range(max_freq_width_lobe, K + max_freq_width_lobe):
                # Variável auxiliar para indexar sort_indices.
                aux_k = k - max_freq_width_lobe
                # Multiplica o vetor horizontal no bin k e em torno da posição m pela janela de Hamming temporal.
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] *= hamming_time[i]

                # Reseta o array auxiliar "sort_indices" que guarda os índices das posições originais do vetor horizontal.
                for i in range(time_width):
                        sort_indices[aux_k, i] = i

                # Ordena o vetor horizontal. Os índices originais são salvos para que a multiplicação pela janela possa ser desfeita posteriormente.
                for i_sort in range(1, time_width):
                    key = X[p, k, m - time_width_lobe + i_sort]
                    j_sort = i_sort - 1
                    while j_sort >= 0 and key < X[p, k, m - time_width_lobe + j_sort]:
                        X[p, k, m - time_width_lobe + j_sort + 1] = X[p, k, m - time_width_lobe + j_sort]
                        sort_indices[aux_k, j_sort + 1] = sort_indices[aux_k, j_sort]
                        j_sort = j_sort - 1
                    X[p, k, m - time_width_lobe + j_sort + 1] = key
                    sort_indices[aux_k, j_sort + 1] = i_sort

            ################################### }

            IF DEBUGTIMER:  
                time_ff = clock()
                timer_sort += (<double> (time_ff - time_ii) ) / CLOCKS_PER_SEC 

            IF DEBUGPRINT:
                pass
                print("Multiplicou pelo Hamming no tempo e ordenou.") #DEBUGPRINT
                print_arr(X[p], [0, K + 2*max_freq_width_lobe, m - time_width_lobe, m + time_width_lobe + 1], colorama.Back.MAGENTA) #DEBUGPRINT

            ################################### Combinação dos vetores ordenados {

            # Itera pelos slices de frequência.    
            for k in range(max_freq_width_lobe, K + max_freq_width_lobe):

                IF DEBUGTIMER:
                    time_ii = clock()

                # Itera pelas posições da janela atual (slice de frequência x segmento temporal), multiplicando pela janela de Hamming na frequência.
                for i in range(time_width):
                    for j in range(freq_width_sparsity):
                        X[p, k - freq_width_sparsity_lobe + j, m - time_width_lobe + i] *= hamming_freq_sparsity[j]
                        
                IF DEBUGPRINT:
                    pass
                    print("Multiplicou pelo Hamming na frequência.") #DEBUGPRINT
                    print_arr(X[p], [k - freq_width_energy_lobe, k + freq_width_sparsity_lobe + 1, m - time_width_lobe, m + time_width_lobe + 1], colorama.Back.RED) #DEBUGPRINT


                # Inicializa o heap com o primeiro elemento de cada array.
                for i in range(num_vectors):
                    # Adiciona o elemento no final da heap.
                    heap_elements[i] = X[p, k - freq_width_sparsity_lobe + i, m - time_width_lobe]
                    
                    heap_origins[i] = i

                    # Inicializa com 0 a posição correspondente do vetor de índices
                    array_indices[i] = 0

                    # Realiza a restauração da heap (heapify_up) após a inserção.
                    j = i
                    j_parent = (j - 1) // 2
                    while j_parent >= 0 and heap_elements[j_parent] > heap_elements[j]:
                        heap_elements[j_parent], heap_elements[j] = heap_elements[j], heap_elements[j_parent]
                        heap_origins[j_parent], heap_origins[j] = i, heap_origins[j_parent]

                        j = j_parent
                        j_parent = (j - 1) // 2

                # Constrói o vetor combinado, utilizando a heap como fila.
                for o in range(combined_size):

                    # Passa o elemento do topo do heap para o vetor combinado.
                    combined[o] = heap_elements[0]

                    # Incrementa a posição de array_indices correspondente ao vetor original do elemento.
                    element_origin = heap_origins[0]
                    array_indices[element_origin] += 1
                    origin_index = array_indices[element_origin]

                    # Coloca o próximo elemento do vetor original no topo do heap. Caso o vetor original já tenha se esgotado, coloca um valor infinito.
                    if origin_index >= len_vectors:
                        heap_elements[0] = INFINITY
                    else:
                        heap_elements[0] = X[p, k - freq_width_sparsity_lobe + element_origin, m - time_width_lobe + origin_index]

                    # Realiza a restauração da heap (heapify_down) após a substituição.
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
                        
                    #print(f"k={k}\ncombined={combined_ndarray[0:k+1]}\nheap_elements={heap_elements_ndarray}\nheap_origins={heap_origins_ndarray}\narray_indices={array_indices_ndarray}\n\n\n") #DEBUGPRINT

                ################################### }

                IF DEBUGTIMER:  
                    time_ff = clock()
                    timer_comb += (<double> (time_ff - time_ii) ) / CLOCKS_PER_SEC 
                

                IF DEBUGPRINT:
                    pass
                    print("combined =", combined_ndarray)

                ################################### Cálculo do índice de Gini {

                arr_norm = 0.0
                gini = 0.0
                
                combined_size_db = <double> combined_size
                for i in range(combined_size):
                    arr_norm = arr_norm + combined[i]
                    gini = gini - 2*combined[i] * (combined_size - i - 0.5)/combined_size_db

                gini = 1 + gini/(arr_norm + epsilon)

                # Índice para a matriz de esparsidade local deve ser ajustado porque essa não tem zero-padding.
                sparsity[p, k - max_freq_width_lobe, m - time_width_lobe] += gini

                IF DEBUGPRINT:
                    pass
                    print(f"gini = {gini}", end = "\n\n") #DEBUGPRINT

                ################################### }
                
                # Desfazer a multiplicação pelo Hamming na frequência
                for i in range(time_width):
                    for j in range(freq_width_sparsity):
                        X[p, k - freq_width_sparsity_lobe + j, m - time_width_lobe + i] /= hamming_freq_sparsity[j]

            #print("sort_indices")
            #print_arr(sort_indices)

            # Desfazer a multiplicação pelo Hamming no tempo, levando em consideração a mudança de índices na ordenação.
            for k in range(max_freq_width_lobe, K + max_freq_width_lobe):
                for i in range(time_width):
                    aux_horiz_vector[sort_indices[k - max_freq_width_lobe, i]] = X[p, k, m - time_width_lobe + i] / hamming_time[sort_indices[k - max_freq_width_lobe, i]]
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] = aux_horiz_vector[i]

            IF DEBUGPRINT:
                pass
                print("Desmultiplicou pelo Hamming no tempo e desfez a ordenação.") #DEBUGPRINT
                print_arr(X[p], [0, K + 2*max_freq_width_lobe, m - time_width_lobe, m + time_width_lobe + 1], colorama.Back.BLUE) #DEBUGPRINT

        IF DEBUGPRINT:
            print(f"Sparsity, p = {p}") #DEBUGPRINT
            print_arr(sparsity[p]) #DEBUGPRINT
    
    ################################################### }} Cálculo da Esparsidade Local

    IF DEBUGTIMER:
        time_f = clock()
        timer_sparsity = (<double> (time_f - time_i) ) / CLOCKS_PER_SEC 
    
    IF DEBUGPRINT:
        print("\n\n ==== Cálculo da energia ====", end="\n\n\n")

    
    IF DEBUGTIMER:
        time_i = clock()
    
    ################################################### Cálculo da Energia Local {{

    # Itera pelos espectrogramas.
    for p in range(P):
        # Itera pelos segmentos temporais.
        for m in range(time_width_lobe, M + time_width_lobe):
            # Itera pelos bins de frequência
            for k in range(max_freq_width_lobe, K + max_freq_width_lobe):
                # Multiplica o vetor horizontal no bin k e em torno da posição m pela janela de Hamming temporal assimétrica.
                for i in range(time_width_lobe + 1):
                    X[p, k, m - time_width_lobe + i] *= hamming_asym_time[i]

            IF DEBUGPRINT:
                pass
                print(f"Multiplicou pelo Hamming no Tempo. p = {p}, m = {m}")
                print_arr(X[p], [0, K + 2*max_freq_width_lobe, m - time_width_lobe, m + 1], colorama.Back.MAGENTA)

            # Itera pelos slices de frequência
            for k in range(max_freq_width_lobe, K + max_freq_width_lobe):
                # Itera pelas posições da janela atual (slice de frequência x segmento temporal), multiplicando pelo Hamming na frequência e calculando a energia.
                for i in range(time_width_lobe + 1):
                    for j in range(freq_width_energy):
                        # Índice para a matriz de energia local deve ser ajustado porque essa não tem zero-padding.
                        energy[p, k - max_freq_width_lobe, m - time_width_lobe] += X[p, k - freq_width_energy_lobe + j, m - time_width_lobe + i] * hamming_freq_energy[j]

                IF DEBUGPRINT:
                    pass
                    print(f"Multiplicou pelo Hamming na frequência e calculou a energia. p = {p}, m = {m}, k = {k}. Array de energia:")
                    print_arr(energy[p], [k - max_freq_width_lobe, 
                                          k - max_freq_width_lobe + 1, 
                                          m - time_width_lobe, 
                                          m - time_width_lobe + 1], colorama.Back.RED)

            # Desfazer a multiplicação pelo Hamming no tempo.
            for k in range(max_freq_width_lobe, K + max_freq_width_lobe):
                for i in range(time_width_lobe + 1):
                    X[p, k, m - time_width_lobe + i] /= hamming_asym_time[i]

    ################################################### }} Cálculo da Energia Local

    IF DEBUGTIMER:
        time_f = clock()
        timer_energy = (<double> (time_f - time_i) ) / CLOCKS_PER_SEC 
        print(f"Timer sparsity: {timer_sparsity}\n\t- Timer sort: {timer_sort}\n\t- Timer comb: {timer_comb}\nTimer energy: {timer_energy}")

    ################################################### Combinação por Esparsidade Local e compensação por Energia Local {{
        
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

        IF DEBUGPRINT:
            print("Combinação para o Smoothed LS.\n")

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


    ################################################### }} Combinação por Esparsidade Local e compensação por Energia Local

    IF DEBUGPRINT:
        print("\n\nEsparsidade local final")
        for p in range(P):
            print(f"p = {p}")
            print_arr(sparsity[p,:,:])

        print("\nEnergia local final")
        for p in range(P):
            print(f"p = {p}")
            print_arr(energy[p,:,:])

    return result_ndarray