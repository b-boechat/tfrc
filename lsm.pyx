import numpy as np
import colorama # Para os prints de debug.
DEF DEBUGTIMER = 0
DEF DEBUGPRINT = 0



IF DEBUGTIMER:
    from timeit import default_timer

cimport cython
from cython.parallel import prange
from libc.math cimport INFINITY

# DEBUGGING

IF DEBUGPRINT:
    def print_arr(arr, color_range = None, color = None):
            colorama.init(autoreset=True)
            I, J = arr.shape
            for i in range(I):
                for j in range(J):
                    if color_range is not None and i >= color_range[0] and i < color_range[1] and j >= color_range[2] and j < color_range[3]:
                        print("{}".format(color + str(arr[i][j])), end="  ")
                    else:
                        print("{}".format(str(arr[i][j])), end="  ")
                print()

            print()


def local_sparsity_wrapper(X, freq_width_energy=41, freq_width_sparsity=17, time_width=13):
    #print(f"freq_width_sparsity = {freq_width_sparsity}\nfreq_width_energy = {freq_width_energy}\ntime_width = {time_width}")
    return local_sparsity(X, freq_width_energy, freq_width_sparsity, time_width)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef local_sparsity(double[:,:,::1] X, Py_ssize_t freq_width_energy, Py_ssize_t freq_width_sparsity, Py_ssize_t time_width):
    
    IF DEBUGTIMER:
        time_init = default_timer()

    cdef:
        Py_ssize_t P = X.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X.shape[1] # Eixo das frequências
        Py_ssize_t M = X.shape[2] # Eixo do tempo

        Py_ssize_t freq_width_energy_lobe = (freq_width_energy-1)//2
        Py_ssize_t freq_width_sparsity_lobe = (freq_width_sparsity-1)//2
        Py_ssize_t time_width_lobe = (time_width-1)//2
        Py_ssize_t p, k, m, i, j, i_sort, j_sort

        double key, max_sparsity, min_energy, energy_sum

        double epsilon = 1e-10

    NUM_THREADS = P # temporário
    
    sparsity_ndarray = np.empty((P, K, M), dtype=np.double)
    energy_ndarray = np.zeros((P, K, M), dtype=np.double) # Matriz de energias precisa ser inicializa com zeros pois é calculada incrementalmente.
    result_ndarray = np.empty((K, M), dtype=np.double)
    
    cdef double[:, :, :] sparsity = sparsity_ndarray
    cdef double[:, :, :] energy = energy_ndarray
    cdef double[:, :] result = result_ndarray

    # Calcula as janelas de Hamming utilizadas no algoritmo, separadamente para cada eixo.
    hamming_freq_energy_ndarray = np.hamming(freq_width_energy)
    hamming_freq_sparsity_ndarray = np.hamming(freq_width_sparsity)
    hamming_time_ndarray = np.hamming(time_width)
    hamming_asym_time_ndarray = np.hamming(time_width * 2 - 1)[0:time_width]

    # DEBUG
    hamming_freq_sparsity_ndarray = 2*np.ones(freq_width_sparsity)
    hamming_time_ndarray = 3*np.ones(time_width)
    hamming_freq_energy_ndarray = 2*np.ones(freq_width_energy)
    hamming_asym_time_ndarray = 10*np.ones(time_width)


    cdef double[:] hamming_freq_energy = hamming_freq_energy_ndarray
    cdef double[:] hamming_freq_sparsity = hamming_freq_sparsity_ndarray
    cdef double[:] hamming_asym_time = hamming_asym_time_ndarray
    cdef double[:] hamming_time = hamming_time_ndarray

    sort_indices_ndarray = np.empty((NUM_THREADS, K, time_width), dtype=np.intp)
    cdef Py_ssize_t[:,:,:] sort_indices = sort_indices_ndarray

    aux_horiz_vector_ndarray = np.empty((NUM_THREADS, time_width), dtype=np.double)
    cdef double[:, :] aux_horiz_vector = aux_horiz_vector_ndarray
    

    ######### {Variáveis para a combinação de vetores ordenados por heap e cálculo do índice de Gini
    cdef:
        Py_ssize_t num_vectors = freq_width_sparsity
        Py_ssize_t len_vectors = time_width
        Py_ssize_t combined_size = num_vectors * len_vectors
        Py_ssize_t j_parent, j_left_child, j_right_child, j_smaller_child, o
        Py_ssize_t element_origin, origin_index


    heap_elements_ndarray = np.empty((NUM_THREADS, num_vectors), dtype=np.double)
    heap_origins_ndarray = np.empty((NUM_THREADS, num_vectors), dtype=np.intp)
    array_indices_ndarray = np.empty((NUM_THREADS, num_vectors), dtype=np.intp)
    combined_ndarray = np.empty((NUM_THREADS, combined_size), dtype=np.double)

    cdef double[:, :] heap_elements = heap_elements_ndarray # Heap que armazena o menor elemento não "consumido" de cada vetor.
    cdef Py_ssize_t[:, :] heap_origins = heap_origins_ndarray # Armazena o vetor de origem do elemento correspondente da heap.
    cdef Py_ssize_t[:, :] array_indices = array_indices_ndarray # Armazena o índice atual na heap de cada vetor.
    cdef double[:, :] combined = combined_ndarray # Armazena o vetor combinado de todos os elementos

    cdef double combined_size_db, arr_norm, gini

    ######### }

    IF DEBUGTIMER:
        timer_sparsity = 0


    ################################################### Cálculo da Esparsidade Local {{ 

    # Itera pelos espectrogramas.
    for p in range(P):#, nogil=True, num_threads=NUM_THREADS):
        # Itera pelos segmentos temporais.
        for m in range(time_width_lobe, M - time_width_lobe):
            
            IF DEBUGTIMER:
                time_i = default_timer() 

            # Itera pelos bins de frequência
            for k in range(K):
                # Multiplica o vetor horizontal no bin k e em torno da posição m pela janela de Hamming temporal.
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] *= hamming_time[i]

                # Reseta o array auxiliar "sort_indices" que guarda os índices das posições originais do vetor horizontal.
                for i in range(time_width):
                        sort_indices[p, k, i] = i

                # Ordena o vetor horizontal. Os índices originais são salvos para que a multiplicação pela janela possa ser desfeita posteriormente.
                for i_sort in range(1, time_width):
                    key = X[p, k, m - time_width_lobe + i_sort]
                    j_sort = i_sort - 1
                    while j_sort >= 0 and key < X[p, k, m - time_width_lobe + j_sort]:
                        X[p, k, m - time_width_lobe + j_sort + 1] = X[p, k, m - time_width_lobe + j_sort]
                        sort_indices[p, k, j_sort + 1] = sort_indices[p, k, j_sort]
                        j_sort = j_sort - 1
                    X[p, k, m - time_width_lobe + j_sort + 1] = key
                    sort_indices[p, k, j_sort + 1] = i_sort

            IF DEBUGTIMER:
                timer_sort_vecs = default_timer() - time_i 
            IF DEBUGPRINT:
                print("Multiplicou pelo Hamming no tempo e ordenou.") 
                print_arr(X[p], [0, K, m - time_width_lobe, m + time_width_lobe + 1], colorama.Back.MAGENTA) 
            

            # Itera pelos slices de frequência.    
            for k in range(freq_width_sparsity_lobe, K - freq_width_sparsity_lobe):
                # Iterar pelas posições da janela atual (slice de frequência x segmento temporal), multiplicando pela janela de Hamming na frequência.
                for i in range(time_width):
                    for j in range(freq_width_sparsity):
                        X[p, k - freq_width_sparsity_lobe + j, m - time_width_lobe + i] *= hamming_freq_sparsity[j]

                IF DEBUGPRINT: 
                    print("Multiplicou pelo Hamming na frequência.") 
                    print_arr(X[p], [k - freq_width_energy_lobe, k + freq_width_sparsity_lobe + 1, m - time_width_lobe, m + time_width_lobe + 1], colorama.Back.RED) 

                IF DEBUGTIMER: 
                    time_i = default_timer()


                ################################### Combinação dos vetores ordenados {

                #arrs = X[p, k-freq_width_sparsity_lobe:k+freq_width_sparsity_lobe+1, m-time_width_lobe:m+time_width_lobe+1]


                # Inicializa o heap com o primeiro elemento de cada array.
                for i in range(num_vectors):
                    # Adiciona o elemento no final da heap.
                    heap_elements[p, i] = X[p, k - freq_width_sparsity_lobe + i, m - time_width_lobe]
                    
                    heap_origins[p, i] = i

                    # Inicializa com 0 a posição correspondente do vetor de índices
                    array_indices[p, i] = 0

                    # Realiza a restauração da heap (heapify_up) após a inserção.
                    j = i
                    j_parent = (j - 1) // 2
                    while j_parent >= 0 and heap_elements[p, j_parent] > heap_elements[p, j]:
                        heap_elements[p, j_parent], heap_elements[p, j] = heap_elements[p, j], heap_elements[p, j_parent]
                        heap_origins[p, j_parent], heap_origins[p, j] = i, heap_origins[p, j_parent]

                        j = j_parent
                        j_parent = (j - 1) // 2

                #print(f"heap_elements={heap_elements_ndarray[p]}\nheap_origins={heap_origins_ndarray[p]}\narray_indices={array_indices_ndarray[p]}\n\n\n") 

                # Constrói o vetor combinado, utilizando a heap como fila.
                for o in range(combined_size):

                    # Passa o elemento do topo do heap para o vetor combinado.
                    combined[p, o] = heap_elements[p, 0]

                    # Incrementa a posição de array_indices correspondente ao vetor original do elemento.
                    element_origin = heap_origins[p, 0]
                    array_indices[p, element_origin] = array_indices[p, element_origin] + 1
                    origin_index = array_indices[p, element_origin]

                    # Coloca o próximo elemento do vetor original no topo do heap. Caso o vetor original já tenha se esgotado, coloca um valor infinito.
                    if origin_index >= len_vectors:
                        heap_elements[p, 0] = INFINITY
                    else:
                        heap_elements[p, 0] = X[p, k - freq_width_sparsity_lobe + element_origin, m - time_width_lobe + origin_index]

                    # Realiza a restauração da heap (heapify_down) após a substituição.
                    j = 0
                    j_left_child = 2*j + 1
                    while j_left_child < num_vectors:
                        j_smaller_child = j_left_child
                        j_right_child = j_left_child + 1
                        if j_right_child < num_vectors and heap_elements[p, j_right_child] < heap_elements[p, j_left_child]:
                            j_smaller_child = j_right_child

                        if heap_elements[p, j] <= heap_elements[p, j_smaller_child]:
                            break
                        
                        heap_elements[p, j], heap_elements[p, j_smaller_child] = heap_elements[p, j_smaller_child], heap_elements[p, j]
                        heap_origins[p, j], heap_origins[p, j_smaller_child] = heap_origins[p, j_smaller_child], heap_origins[p, j]

                        j = j_smaller_child
                        j_left_child = 2*j + 1
                        
                    #print(f"o={o}\ncombined={combined_ndarray[p,0:o+1]}\nheap_elements={heap_elements_ndarray[p]}\nheap_origins={heap_origins_ndarray[p]}\narray_indices={array_indices_ndarray[p]}\n\n\n") 

                ################################### }

                IF DEBUGPRINT:
                    print("combined =", combined_ndarray[p])

                ################################### Cálculo do índice de Gini {

                arr_norm = 0.0
                gini = 0.0
                
                combined_size_db = <double> combined_size
                for i in range(combined_size):
                    arr_norm = arr_norm + combined[p, i]
                    gini = gini - 2*combined[p, i] * (combined_size - i - 0.5)/combined_size_db

                gini = 1 + gini/(arr_norm + epsilon)

                sparsity[p, k, m] = gini

                IF DEBUGPRINT:
                    print(f"gini = {gini}", end="\n\n") 

                ################################### }
                
                IF DEBUGTIMER: 
                    timer_sparsity += default_timer() - time_i 



                # Desfazer a multiplicação pelo Hamming na frequência
                for i in range(time_width):
                    for j in range(freq_width_sparsity):
                        X[p, k - freq_width_sparsity_lobe + j, m - time_width_lobe + i] /= hamming_freq_sparsity[j]


            # Desfazer a multiplicação pelo Hamming no tempo, levando em consideração a mudança de índices na ordenação.
            for k in range(K):
                for i in range(time_width):
                    aux_horiz_vector[p, sort_indices[p, k, i]] = X[p, k, m - time_width_lobe + i] / hamming_time[sort_indices[p, k, i]]
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] = aux_horiz_vector[p, i]

            IF DEBUGPRINT:
                print("Desmultiplicou pelo Hamming no tempo e desfez a ordenação.") 
                print_arr(X[p], [0, K, m - time_width_lobe, m + time_width_lobe + 1], colorama.Back.BLUE) 

        IF DEBUGPRINT:
            print("Sparsity") 
            print_arr(sparsity[p])
    
    ################################################### }} Cálculo da Esparsidade Local


    ################################################### Combinação por Esparsidade Local {{

    # Em cada ponto, seleciona para o espectrograma combinado o valor do espectrograma de menor esparsidade local.
    for k in range(K):
        for m in range(M):
            max_sparsity = -1.0
            for p in range(P):
                if sparsity[p, k, m] > max_sparsity:
                    result[k, m] = X[p, k, m]
                    max_sparsity = sparsity[p, k, m]



    ################################################### }} Combinação por Esparsidade Local

    IF DEBUGTIMER:
        time_i = default_timer() 

    IF DEBUGPRINT:
        print("\n\n ==== Cálculo da energia ====", end="\n\n\n")

    # Itera pelos espectrogramas.
    for p in range(P):
        # Itera pelos segmentos temporais.
        for m in range(time_width_lobe, M - time_width_lobe):
            # Itera pelos bins de frequência
            for k in range(K):
                # Multiplica o vetor horizontal no bin k e em torno da posição m pela janela de Hamming temporal assimétrica.
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] *= hamming_asym_time[i]

            IF DEBUGPRINT:
                print("Multiplicou pelo Hamming no Tempo.")
                print_arr(X[p], [0, K, m - time_width_lobe, m + time_width_lobe + 1], colorama.Back.MAGENTA)

            # Itera pelos slices de frequência
            for k in range(freq_width_energy_lobe, K - freq_width_energy_lobe):
                # Itera pelas posições da janela atual (slice de frequência x segmento temporal), multiplicando pelo Hamming na frequência e calculando a energia.
                for i in range(time_width):
                    for j in range(freq_width_energy):
                        energy[p, k, m] += X[p, k - freq_width_energy_lobe + j, m - time_width_lobe + i] * hamming_freq_energy[j]
                IF DEBUGPRINT:
                    print("Multiplicou pelo Hamming na frequência e calculou a energia.")
                    print_arr(energy[p], [k - freq_width_energy_lobe, k + freq_width_energy_lobe + 1, m - time_width_lobe, m + time_width_lobe + 1], colorama.Back.RED)

            # Desfazer a multiplicação pelo Hamming no tempo.
            for k in range(K):
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] /= hamming_asym_time[i]

    IF DEBUGTIMER:
        timer_energy = default_timer() - time_i 

    # Realiza a compensação de energia.
    for k in range(freq_width_energy_lobe, K - freq_width_energy_lobe): # TODO ajeitar nas bordas.
        for m in range(time_width_lobe, M - time_width_lobe):
            min_energy = INFINITY
            energy_sum = epsilon
            for p in range(P):
                energy_sum += energy[p, k, m]
                if energy[p, k, m] < min_energy:
                    min_energy = energy[p, k, m] 
            result[k, m] *= min_energy/energy_sum

    IF DEBUGPRINT:
        print("Combinação final.")
        print_arr(result)

    if DEBUGTIMER:
        timer_total = default_timer() - time_init
        print(f"\tSort vectors = {timer_sort_vecs}\n\tCombine and sparsity = {timer_sparsity}\n\tEnergy = {timer_energy}\n\tTotal = {timer_total}") 

    return result_ndarray