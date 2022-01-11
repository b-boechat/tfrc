import numpy as np
from timeit import default_timer # DEBUG APENAS

cimport cython
from cython.parallel import prange
from libc.math cimport INFINITY

# DEBUGGING
def print_arr(arr):
        M, N = arr.shape
        for m in range(M):
            for n in range(N):
                print("{}".format(arr[m][n]), end="  ")
            print()

        print("\n\n\n")


def local_sparsity(double[:,:,:] X, Py_ssize_t freq_width_energy=41, Py_ssize_t freq_width_sparsity=17, Py_ssize_t time_width=13):
    cdef:
        Py_ssize_t P = X.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X.shape[1] # Eixo das frequências
        Py_ssize_t M = X.shape[2] # Eixo do tempo

        Py_ssize_t freq_width_energy_lobe = (freq_width_energy-1)//2
        Py_ssize_t freq_width_sparsity_lobe = (freq_width_sparsity-1)//2
        Py_ssize_t time_width_lobe = (time_width-1)//2
        Py_ssize_t p, k, m, i, j, i_sort, j_sort

        double key, max_sparsity, min_energy, energy_sum

    
    time_init = default_timer()

    sparsity_ndarray = np.empty((P, K, M), dtype=np.double)
    energy_ndarray = np.zeros((P, K, M), dtype=np.double) # Matriz de energias precisa ser inicializa com 0 pois é calculada incrementalmente.
    result_ndarray = np.empty((K, M), dtype=np.double)
    
    cdef double[:, :, :] sparsity = sparsity_ndarray
    cdef double[:, :, :] energy = energy_ndarray
    cdef double[:, :] result = result_ndarray

    # Calculate hamming windows on both axises.

    hamming_freq_energy_ndarray = np.hamming(freq_width_energy)
    hamming_freq_sparsity_ndarray = np.hamming(freq_width_sparsity)
    hamming_time_ndarray = np.hamming(time_width)
    hamming_asym_time_ndarray = np.hamming(time_width * 2 - 1)[0:time_width]

    # DEBUG
    #hamming_freq_sparsity_ndarray = 2*np.ones(freq_width_sparsity)
    #hamming_time_ndarray = 3*np.ones(time_width)
    #hamming_freq_energy_ndarray = 2*np.ones(freq_width_energy)
    #hamming_asym_time_ndarray = 10*np.ones(time_width)


    cdef double[:] hamming_freq_energy = hamming_freq_energy_ndarray
    cdef double[:] hamming_freq_sparsity = hamming_freq_sparsity_ndarray
    cdef double[:] hamming_asym_time = hamming_asym_time_ndarray
    cdef double[:] hamming_time = hamming_time_ndarray

    sort_indices_ndarray = np.empty((K, time_width), dtype=np.intp)
    cdef Py_ssize_t[:,:] sort_indices = sort_indices_ndarray

    aux_horiz_vector_ndarray = np.empty(time_width, dtype=np.double)
    cdef aux_horiz_vector = aux_horiz_vector_ndarray


    time_sparsity = 0 #DEBUGTIMER

    # Itera pelos espectrogramas.
    for p in range(P):
        # Itera pelos segmentos temporais.
        for m in range(time_width_lobe, M - time_width_lobe):

            # Itera pelos bins de frequência
            for k in range(K):
                # Multiplica o vetor horizontal no bin k e em torno da posição m pela janela de Hamming temporal.
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] *= hamming_time[i]

                # Reseta o array auxiliar "sort_indices" que guarda os índices das posições originais do vetor horizontal.
                for i in range(time_width):
                        sort_indices[k, i] = i

                # Ordena o vetor horizontal. Os índices originais são salvos para que a multiplicação pela janela possa ser desfeita posteriormente.
                for i_sort in range(1, time_width):
                    key = X[p, k, m - time_width_lobe + i_sort]
                    j_sort = i_sort - 1
                    while j_sort >= 0 and key < X[p, k, m - time_width_lobe + j_sort]:
                        X[p, k, m - time_width_lobe + j_sort + 1] = X[p, k, m - time_width_lobe + j_sort]
                        sort_indices[k, j_sort + 1] = sort_indices[k, j_sort]
                        j_sort -= 1
                    X[p, k, m - time_width_lobe + j_sort + 1] = key
                    sort_indices[k, j_sort + 1] = i_sort

            #print("Multiplicou pelo Hamming no Tempo e ordenou")
            #print_arr(X[p])
            

            # Itera pelos slices de frequência.    
            for k in range(freq_width_sparsity_lobe, K - freq_width_sparsity_lobe):
                # Iterar pelas posições da janela atual (slice de frequência x segmento temporal), multiplicando pela janela de Hamming na frequência.
                for i in range(time_width):
                    for j in range(freq_width_sparsity):
                        X[p, k - freq_width_sparsity_lobe + j, m - time_width_lobe + i] *= hamming_freq_sparsity[j]
                        
                #print("Multiplicou pelo Hamming na Freq")
                #print_arr(X[p])

                time_i = default_timer() #DEBUGTIMER
                
                # Calcular a esparsidade para a janela atual.
                sparsity[p, k, m] = _calculate_gini_from_sorted_arrays(X[p, k-freq_width_sparsity_lobe:k+freq_width_sparsity_lobe+1, m-time_width_lobe:m+time_width_lobe+1
                                                            ], 1, freq_width_sparsity, time_width)
                time_f = default_timer() #DEBUGTIMER
                time_sparsity += time_f - time_i #DEBUGTIMER



                # Desfazer a multiplicação pelo Hamming na frequência
                for i in range(time_width):
                    for j in range(freq_width_sparsity):
                        X[p, k - freq_width_sparsity_lobe + j, m - time_width_lobe + i] /= hamming_freq_sparsity[j]

            #print("sort_indices")
            #print_arr(sort_indices)

            # Desfazer a multiplicação pelo Hamming no tempo, levando em consideração a mudança de índices na ordenação.
            for k in range(K):
                for i in range(time_width):
                    aux_horiz_vector[sort_indices[k, i]] = X[p, k, m - time_width_lobe + i] / hamming_time[sort_indices[k, i]]
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] = aux_horiz_vector[i]

            #print("Desmultiplicou pelo Hamming no Tempo")
            #print_arr(X[p])

        #print("Sparsity")
        #print_arr(sparsity[p])
    

    # Em cada ponto, seleciona para o espectrograma combinado o valor do espectrograma de menor esparsidade local.
    for k in range(K):
        for m in range(M):
            max_sparsity = -1.0
            for p in range(P):
                if sparsity[p, k, m] > max_sparsity:
                    result[k, m] = X[p, k, m]
                    max_sparsity = sparsity[p, k, m]


    time_i_energy = default_timer() #DEBUGTIMER

    # Itera pelos espectrogramas.
    for p in range(P):
        # Itera pelos segmentos temporais.
        for m in range(time_width_lobe, M - time_width_lobe):
            # Itera pelos bins de frequência
            for k in range(K):
                # Multiplica o vetor horizontal no bin k e em torno da posição m pela janela de Hamming temporal assimétrica.
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] *= hamming_asym_time[i]

            #print("Multiplicou pelo Hamming no Tempo.")
            #print_arr(X[p])

            # Itera pelos slices de frequência
            for k in range(freq_width_energy_lobe, K - freq_width_energy_lobe):
                # Itera pelas posições da janela atual (slice de frequência x segmento temporal), multiplicando pelo Hamming na frequência e calculando a energia.
                for i in range(time_width):
                    for j in range(freq_width_energy):
                        energy[p, k, m] += X[p, k - freq_width_energy_lobe + j, m - time_width_lobe + i] * hamming_freq_energy[j]

            #print("Multiplicou pelo Hamming na frequência e calculou a energia.")
            #print_arr(energy[p])

            # Desfazer a multiplicação pelo Hamming no tempo.
            for k in range(K):
                for i in range(time_width):
                    X[p, k, m - time_width_lobe + i] /= hamming_asym_time[i]
                    
    time_f_energy = default_timer() #DEBUGTIMER

    # Realiza a compensação de energia.
    for k in range(freq_width_energy_lobe, K - freq_width_energy_lobe): # TODO ajeitar nas bordas.
        for m in range(time_width_lobe, M - time_width_lobe):
            min_energy = INFINITY
            energy_sum = 0.0
            for p in range(P):
                energy_sum += energy[p, k, m]
                if energy[p, k, m] < min_energy:
                    min_energy = energy[p, k, m] 
            result[k, m] *= min_energy/energy_sum

    #for p in range(P):        
    #    print(f"Sparsity, p={p}.")
    #    print_arr(sparsity[p])
    
    #for p in range(P):        
    #    print(f"Energy, p={p}.")
    #    print_arr(energy[p])

    #print("Resultado final.")
    #print_arr(result)

    time_final = default_timer() #DEBUGTIMER
    
    print(f"\tSparsity = {time_sparsity}\n\tEnergy = {time_f_energy - time_i_energy}\n\tTotal = {time_final - time_init}") #DEBUGTIMER

    return result_ndarray





def _calculate_gini_from_sorted_arrays(double[:, :] arrs not None, int vecs_axis, Py_ssize_t size0, Py_ssize_t size1):
    """
    Recebe uma matriz composta por vetores ordenados, combina-os em um único vetor ordenado e calcula o coeficiente de Gini para este.

    :param arrs: Matriz de vetores ordenados.
    :param vecs_axis: Eixo ao longo do qual a matriz está ordenada. 0: arrs[:, x] é ordenado. 1: arrs[x, :] é ordenado.
    :param size0, size1: Dimensões da matriz (size0 x size1)
    :return: Coeficiente de Gini calculado a partir do vetor ordenado contendo todos os elementos da matriz.
    """

    cdef Py_ssize_t result_size = size0*size1
    cdef Py_ssize_t num_vectors
    cdef Py_ssize_t i, j, k, mirror_i
    cdef double gini

    # Dois vetores são usados para armazenar os resultados, alternando nas iterações 
    # lido em result_even e escrito em result_odd -> lido em result_odd e escrito em result_even -> repete
    cdef result_odd = np.zeros(result_size, dtype=np.double)
    cdef result_even = np.zeros(result_size, dtype=np.double)

    cdef Py_ssize_t[:] pos_read
    cdef Py_ssize_t[:] pos_write

    cdef double[:] result_write = result_odd
    cdef double[:] result_read = result_even

    if vecs_axis == 1: # Vetores horizontais (arrs[x, :]) vão ser combinados ao longo das linhas.
        num_vectors = size0

        # Copia a matriz para vetor de resultados de iterações pares, em sequência de vetores horizontais.
        for i in range(size0):
            for j in range(size1):
                result_read[i*size1+j] = arrs[i, j]

        # Inicializa o vetor de posições para iterações pares
        pos_even = size1 * np.arange(num_vectors + 1, dtype=np.intp)

    else: # Vetores verticais (arrs[:, x]) vão ser combinado ao longo das colunas.

        num_vectors = size1
        # Copia a matriz para vetor de resultados de iterações pares, em sequência de vetores verticais.
        for i in range(size1):
            for j in range(size0):
                result_read[i*size0+j] = arrs[j, i]

        # Inicializa o vetor de posições para iterações pares
        pos_even = size0 * np.arange(num_vectors + 1, dtype=np.intp)

    # Cria o vetor de posição dos vetores para iterações ímpares
    pos_odd = np.zeros( (num_vectors + 1)//2 + 1, dtype=np.intp)
    # Inicializa a primeira posição como 0, já que o primeiro vetor sempre estará no início de result_odd.
    pos_odd[0] = 0


    k = 0
    while True:

        k += 1
        if k % 2 == 1:
            # k ímpar: lê no even, escreve no odd
            pos_read = pos_even 
            pos_write = pos_odd
            
            result_read = result_even
            result_write = result_odd
        else:
            # k par: lê no odd, escreve no even
            pos_read = pos_odd
            pos_write = pos_even
            
            result_read = result_odd
            result_write = result_even


        next_num_vectors = (num_vectors + 1) // 2

        #print(f"\n\nk={k}, num_vectors={num_vectors}, next_num_vectors={next_num_vectors}\n")     
        pos_write[next_num_vectors] = result_size
    
        for i in range(next_num_vectors):
            # Índice do vetor que vai ser combinado com i
            mirror_i = num_vectors - 1 - i

            #print(f"i={i}, mirror_i={mirror_i}")

            if i > 0:
                pos_write[i] = pos_write[i-1]   +   pos_read[i]-pos_read[i-1]   +   pos_read[mirror_i+2]-pos_read[mirror_i+1]

            #print(f"pos_even={pos_even}")
            #print(f"pos_odd={pos_odd}")

            if i == mirror_i:
                #print("Single!")
                #print(f"\tarr={list(result_read[pos_read[i]:pos_read[i+1]])}")
                result_write[pos_write[i]:pos_write[i+1]] = result_read[pos_read[i]:pos_read[i+1]] 

            else:
                #print("Merge!")
                #print(f"\tarr1={list(result_read[pos_read[i]:pos_read[i+1]])}")
                #print(f"\tarr2={list(result_read[pos_read[mirror_i]:pos_read[mirror_i+1]])}")
                #print(f"\tN1={pos_read[i+1] - pos_read[i]}")
                #print(f"\tN2={pos_read[mirror_i+1] - pos_read[mirror_i]}")
                #print(f"\tpos_write[i]={pos_write[i]}")
                _merge_two_sorted_arrays(result_read[pos_read[i]:pos_read[i+1]], # arr1
                                            result_read[pos_read[mirror_i]:pos_read[mirror_i+1]], # arr2
                                            pos_read[i+1] - pos_read[i], # N1
                                            pos_read[mirror_i+1] - pos_read[mirror_i], # N2
                                            result_write, # result
                                            pos_write[i]
                )
            
            #print(f"result_odd={result_odd}")
            #print(f"result_even={result_even}\n\n")

        num_vectors = next_num_vectors

        if num_vectors == 1:
            if k % 2 == 1:
                gini = calculate_gini_pre_sorted(result_odd, result_size)
            else:
                gini = calculate_gini_pre_sorted(result_even, result_size)

            return gini

@cython.boundscheck(False)
@cython.wraparound(False) 
def calculate_gini_pre_sorted(double[:] arr, Py_ssize_t N):
    """
    Calcula o índice de Gini para o array ordenado fornecido.
    A ordenação não é checada. Se o array não estiver ordenado, o resultado será errado.
    :param arr: Array ordenado para o qual será calculado o índice de Gini
    :param N: Tamanho de arr.
    :return: Índice de Gini calculado.
    """
    cdef:
        double epsilon = 0.00001
        double gini = 0.0
        double arr_norm = 0.0
        Py_ssize_t i
    
    for i in prange(N, nogil=True):
        arr_norm += arr[i]
        gini -= 2*arr[i] * (N - 1./2. - i )/N
    
    gini = 1 + gini/(arr_norm + epsilon)

    return gini


@cython.boundscheck(False)
@cython.wraparound(False) 
def _merge_two_sorted_arrays(double[:] arr1 not None, double[:] arr2 not None, Py_ssize_t N1, Py_ssize_t N2, double[:] result not None, Py_ssize_t pos):
    """
    Combina dois arrays ordenados arr1 e arr2 em um array ordenado, mantendo a ordenação.
    O resultado é armazenado no array result, a partir da posição fornecida. Esse array deve ter espaço o bastante para esse armazenamento

    A função tem comportamento não determinado se algum dos arrays não estiver ordenado.

    :param arr1: Primeiro array a ser combinado.
    :param arr2: Segundo array a ser combinado.
    :param N1, N2: Tamanhos dos arrays arr1 e arr2, respectivamente. Passados para a função para melhora de performance.
    :param result: Array no qual vai ser armazenado o resultado. Ele deve ter N1 + N2 posições disponíveis a partir de pos.
    :param pos: Posição a partir da qual o array result será sobrescrito.
    """
    cdef :
        Py_ssize_t i = 0 # Indexa arr1
        Py_ssize_t j = 0 # Indexa arr2
        Py_ssize_t k = pos # Indexa result (saída)
        Py_ssize_t N_comb = N1 + N2

    while k < pos + N_comb:    

        # Elementos de arr1 já terminaram
        if i >= N1: 
            # Pop em um elemento de arr2.
            result[k] = arr2[j] 
            j += 1
            k += 1

        # Elementos de arr2 já terminaram ou o elemento atual de arr1 é menor igual que o de arr2.
        elif j >= N2 or arr1[i] <= arr2[j]:

            # Pop em um elemento de arr1.                              
            result[k] = arr1[i] 
            i += 1
            k += 1    

         # Elemento de arr2 é menor que o elemento de arr1.
        else:
            # Pop em um elemento de arr2.
            result[k] = arr2[j] 
            j += 1
            k += 1