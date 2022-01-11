import pyximport; pyximport.install(language_level="3")
from scipy.special import comb
import numpy as np
import librosa
import sortednp as snp

import _test_dev_lsm
import lsm

from timeit import default_timer, timeit

def calculate_gini_pre_sorted(arr, N):
        epsilon = 0.00001
        return 1 - 2 * np.sum(np.multiply(arr, (N - np.arange(1, N + 1) + 1 / 2) / N) ) / (sum(arr) + epsilon)


def getSortMeasure(arr):
    N = len(arr)
    inv_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (arr[i] > arr[j]):
                inv_count += 1
  
    return -2*inv_count/comb(N, 2) + 1


def sort_merge_snp(arr, N):
    N_mid = (N+1)//2
    return snp.merge(np.sort(arr[:N_mid]), np.sort(arr[N_mid:]))


#def _generate_hamming(lines, columns):
#    return np.outer(np.hamming(lines), np.hamming(columns))
    

#def _generate_asym_hamming(lines, columns):
#    hamm = np.outer(np.hamming(lines), np.hamming(columns * 2 - 1))
#    return hamm[:, 0:columns]



def test_merge_sorted_arrays():

    arrs = np.random.randint(0, 30, size=(5, 5)).astype(np.double)

    arrs = np.sort(arrs, axis=0)

    print(arrs)

    print(_test_dev_lsm._merge_sorted_arrays_from_matrix(arrs, 1, arrs.shape[0], arrs.shape[1]))


def test_sort():

    N = 20

    arr = np.random.rand(N).astype(np.double)
    arr = np.multiply(arr, np.hamming(N))
    #print(arr)

    print(f"Primeira metade: {getSortMeasure(arr[0 : (N+1)//2])}")
    print(f"Segunda metade: {getSortMeasure(arr[(N+1)//2 :])}")

    print("Numpy default:")
    print(timeit(lambda: np.sort(arr), number=100))

    print("Numpy mergesort (timsort):")
    print(timeit(lambda: np.sort(arr, kind='mergesort'), number=100))

    print("Sort_for_lsm:")
    print(timeit(lambda: _test_dev_lsm._sort_for_lsm(arr, N), number=100))

    print("Sort_for_lsm_no_merge:")
    print(timeit(lambda: _test_dev_lsm._sort_for_lsm_no_merge(arr, N), number=100))

    print("Sort_merge_snp:")
    print(timeit(lambda: sort_merge_snp(arr, N), number=100))


def test_sort_stft():
    file_path = "audio/Spanish2.wav"
    n_fft = 8192
    hop_length = 128
    win_length = 2048

    ana_window_width = 17
    width_lobe = (ana_window_width - 1) // 2

    width_lobe = int((ana_window_width-1)/2)

    audio_data, sr = librosa.load("Spanish2.wav", sr=None)


    stft = np.square(np.abs(librosa.stft(audio_data, n_fft=n_fft,
                                                    hop_length=hop_length, win_length=win_length,
                                                    )).astype(np.double))

    print(stft.shape)

    timer_cython = 0
    timer_np = 0
    k = 0

    mean_sort_measure_first_half = 0
    mean_sort_measure_second_half = 0

    for i in range(0, 4097):
        for j in range(width_lobe, 19001 - width_lobe - 1):
            arr = stft[i, j - width_lobe : j + width_lobe + 1]
            arr = np.multiply(arr, np.hamming(ana_window_width))
            mean_sort_measure_first_half += getSortMeasure(arr[0 : width_lobe+1])
            mean_sort_measure_second_half += getSortMeasure(arr[width_lobe+1:])

            time_i = default_timer()
            np.sort(arr)
            time_f = default_timer()
            timer_np += time_f - time_i

            time_i = default_timer()
            _test_dev_lsm._sort_for_lsm(arr, ana_window_width)
            time_f = default_timer()
            timer_cython += time_f - time_i

            k += 1

    mean_sort_measure_first_half /= k
    mean_sort_measure_second_half /= k
    print(f"Primeira metade: {mean_sort_measure_first_half}")
    print(f"Primeira metade: {mean_sort_measure_second_half}")

    print(f"timer_np={timer_np/k}\ntimer_cython={timer_cython/k}")



def test_gini():

    arr = np.random.rand(10000000)
    arr = np.random.rand(400)
   
    arr = np.sort(arr).astype(np.double)

    print("Numpy:")
    print(timeit(lambda: print(f"Gini_numpy: {calculate_gini_pre_sorted(arr, len(arr)):.6f}"), number=5))

    print("\n\n\n")

    print("Cython:")
    print(timeit(lambda: print(f"Gini_cython: {_test_dev_lsm.calculate_gini_pre_sorted(arr, len(arr)):.6f}"), number=5))


def test_lsm():

    def print_arr(arr):
        M, N = arr.shape
        for m in range(M):
            for n in range(N):
                print("{}".format(int(arr[m][n])), end="  ")
            print()
        print("\n\n\n")

    arr = np.ones(72).reshape((2, 6, 6)).astype(np.double)
    arr[0, 1, 1] = 5.0
    arr[0, 2, 2] = 4.0

    arr[1, 3, 1] = 7.0
    arr[1, 4, 3] = 6.0

    #arr = np.random.rand(900).reshape((30, 30)).astype(np.double)
    #arr = np.expand_dims(arr, axis=0)

    #print(arr)

    lsm.local_sparsity(arr, freq_width_energy=3, freq_width_sparsity=3, time_width=3)
    




#test_merge_sorted_arrays()
#test_sort()
#test_sort_stft()
#test_gini()
test_lsm()