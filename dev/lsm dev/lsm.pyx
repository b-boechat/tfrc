import numpy as np

cimport cython
from cython.parallel import prange





@cython.boundscheck(False)
@cython.wraparound(False)   
def _sort_for_lsm(double[:] arr not None, Py_ssize_t N):
    """
    Ordena o vetor arr, sem alterar o original. Função auxiliar para o Local Sparsity Method, otimizada para a performance nesse problema específico.
    
    :param arr: Array a ser ordenado.
    :param N: Tamanho de arr.
    :return: Vetor ordenado.
    """

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t N_mid = (N + 1)//2
    cdef double key
    
    cdef double[:] aux = np.empty(N, dtype=np.double)
    aux[:] = arr

    # Ordena a primeira metade do vetor (incluindo o ponto central).
    for i in range(1, N_mid):
        key = aux[i]
        j = i-1
        while j >= 0 and key < aux[j]:
            aux[j+1] = aux[j]
            j -= 1
        aux[j+1] = key

    # Ordena a segunda metade do vetor em ordem reversa.
    for i in range(N-2, N_mid-1, -1):
        key = aux[i]
        j = i+1
        while j < N and key < aux[j]:
            aux[j-1] = aux[j]
            j += 1
        aux[j-1] = key

    result = np.empty(N, dtype=np.double)
    cdef double[:] result_view = result

    i = 0
    j = N - 1
    k = 0

    # Combina as duas metades do vetor em ordem direta.
    while k < N:
        if i >= N_mid:
            result_view[k] = aux[j]
            k += 1
            j -= 1
        
        elif j < N_mid or aux[i] <= aux[j]:
            result_view[k] = aux[i]
            k += 1
            i += 1
        else:
            result_view[k] = aux[j]
            k += 1
            j -= 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False) 
def _merge_sorted_arrays_from_matrix(double[:, :] arrs not None, int vecs_axis, Py_ssize_t size0, Py_ssize_t size1):
    """
    Recebe uma matriz composta por vetores ordenados, e os combina em um único vetor ordenado de saída.

    :param arrs: Matriz de vetores ordenados.
    :param vecs_axis: Eixo ao longo do qual a matriz está ordenada. 0: arrs[:, x] é ordenado. 1: arrs[x, :] é ordenado.
    :param size0, size1: Dimensões da matriz (size0 x size1)
    :return: Vetor ordenado de tamanho size0xsize1 contendo todos os elementos da matriz.
    """

    cdef Py_ssize_t result_size = size0*size1
    cdef Py_ssize_t num_vectors
    cdef Py_ssize_t i, j, k, mirror_i

    # Dois vetores são usados para armazenar os resultados, alternando nas iterações 
    # lido em result_even e escrito em result_odd -> lido em result_odd e escrito em result_even -> repete
    cdef result_odd = np.zeros(result_size, dtype=np.double)
    cdef result_even = np.zeros(result_size, dtype=np.double)

    cdef int[:] pos_read
    cdef int[:] pos_write

    cdef double[:] result_write = result_odd
    cdef double[:] result_read = result_even


    if vecs_axis == 1: # Vetores horizontais (arrs[x, :]) vão ser combinados ao longo das linhas.

        num_vectors = size0

        # Copia a matriz para vetor de resultados de iterações pares, em sequência de vetores horizontais.
        for i in range(size0):
            for j in range(size1):
                result_read[i*size1+j] = arrs[i, j]

        # Inicializa o vetor de posições para iterações pares
        pos_even = size1 * np.arange(num_vectors + 1, dtype=np.intc)

    else: # Vetores verticais (arrs[:, x]) vão ser combinado ao longo das colunas.

        num_vectors = size1

        # Copia a matriz para vetor de resultados de iterações pares, em sequência de vetores verticais.
        for i in range(size1):
            for j in range(size0):
                result_read[i*size0+j] = arrs[j, i]

        # Inicializa o vetor de posições para iterações pares
        pos_even = size0 * np.arange(num_vectors + 1, dtype=np.intc)




    # Cria o vetor de posição dos vetores para iterações ímpares
    pos_odd = np.zeros( (num_vectors + 1)//2 + 1, dtype=np.intc  )
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
                return result_odd
            else:
                return result_even


    

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
    
    N_comb = N1 + N2

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




