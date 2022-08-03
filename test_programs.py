from debug import print_arr
import numpy as np

#import pyximport; pyximport.install(language_level="3")

from lukin_todd import lukin_todd_wrapper
from lukin_todd_v1 import lukin_todd_v1_wrapper
from lukin_todd_baseline import lukin_todd_baseline_wrapper
from lsm import local_sparsity_wrapper
from lsm_baseline import local_sparsity_baseline_wrapper
from feulo_integration import feulo_lukin_todd_wrapper, feulo_spectrogram_local_sparsity_wrapper

def test_method(method):

    def create_3d_arr():
        arr = np.ones(72).reshape(2, 6, 6).astype(np.double)
        arr[0, 0, 0] = 2.00
        arr[0, 0, 1] = 3.00
        arr[0, 0, 2] = 5.0
        arr[0, 1, 0] = 5.0
        arr[0, 1, 1] = 5.0
        arr[0, 1, 2] = 4.00
        arr[0, 2, 0] = 6.00
        arr[0, 2, 1] = 5.0
        arr[0, 2, 2] = 6.00

        arr[1, 0, 0] = 2.00
        arr[1, 0, 1] = 6.00
        arr[1, 0, 2] = 12.00
        arr[1, 2, 0] = 7.00
        arr[1, 2, 1] = 2.0
        arr[1, 2, 2] = 7.00
        
        print("p = 0:") 
        print_arr(arr[0])

        print("p = 1:") 
        print_arr(arr[1])
        return arr

    def create_2d_arr():
        arr = np.ones(25).reshape(5, 5).astype(np.double)
        arr[1, 1] = 5
        arr[1, 0] = 7
        arr[2, 0] = 4
        arr[2, 1] = 8
        arr[2, 2] = 2
        arr[2, 3] = 6
        arr[2, 4] = 5

        arr[3, 0] = 2
        arr[3, 1] = 3
        print_arr(arr)
        arr = np.expand_dims(arr, axis=0)
        return arr

    #arr = create_2d_arr()
    arr = create_3d_arr()

    if method == "lt":
        print("Lukin-Todd")

        print("\n\n\n==========\nCython:\n\n=============\n\n\n")
        result_cython = lukin_todd_wrapper(arr, freq_width=3, time_width=5, eta=8)        
        print_arr(result_cython)
        

        print("\n\n\n==========\nCython old:\n\n=============\n\n\n")
        result_cython = lukin_todd_v1_wrapper(arr, freq_width=3, time_width=5, eta=8)        
        print_arr(result_cython)

        print("\n\n\n==========\nCython baseline:\n\n=============\n\n\n")
        result_cython = lukin_todd_baseline_wrapper(arr, freq_width=3, time_width=5, eta=8)        
        print_arr(result_cython)
        
        print("\n\n\n==========\nFeulo:\n\n=============\n\n\n")
        result_feulo = feulo_lukin_todd_wrapper(arr, freq_width=3, time_width=5, eta=8)   
        print_arr(result_feulo)

    elif method == "lsm":
        print("Local Sparsity")
        print("\n\n\n==========\nCython:\n\n=============\n\n\n")
        result_cython = local_sparsity_wrapper(arr, freq_width_energy=3, freq_width_sparsity=5, time_width=5, zeta=10)        
        print_arr(result_cython) 

        print("\n\n\n==========\nCython baseline:\n\n=============\n\n\n")
        result_cython = local_sparsity_baseline_wrapper(arr, freq_width_energy=3, freq_width_sparsity=5, time_width=5, zeta=10)     
        print_arr(result_cython)

        print("\n\n\n==========\nFeulo:\n\n=============\n\n\n")
        result_feulo = feulo_spectrogram_local_sparsity_wrapper(arr, freq_width_energy=3, freq_width_sparsity=5, time_width=5, zeta=10)   
        print_arr(result_feulo) 

    else:
        print("Especifique o método")

#test_method("lt")
test_method("lsm")