from unicodedata import name
from debug import print_arr, print_arr_interpol
import numpy as np
import pandas as pd
import pickle

#import pyximport; pyximport.install(language_level="3")

from lukin_todd import lukin_todd_wrapper
from lukin_todd_v1 import lukin_todd_v1_wrapper
from lukin_todd_baseline import lukin_todd_baseline_wrapper

from lsm import local_sparsity_wrapper
from lsm_baseline import local_sparsity_baseline_wrapper
from lsm_baseline_interpol import local_sparsity_baseline_interpolation_wrapper

from fls import fast_local_sparsity_wrapper

from swgm import swgm_wrapper

from feulo_integration import feulo_lukin_todd_wrapper, feulo_spectrogram_local_sparsity_wrapper, feulo_fast_local_sparsity_wrapper, feulo_swgm_wrapper

def compare_representations(bkp1_path, bkp2_path, range_i=[1820, 1833], range_j=[24, 35], epsilon = 10e-5):

    with open(bkp1_path, "rb") as input_file:
        analysis1 = pickle.load(input_file)
        arr1 = analysis1.combined_tfr

    with open(bkp2_path, "rb") as input_file:
        analysis2 = pickle.load(input_file)
        arr2 = analysis2.combined_tfr
    
    assert(arr1.shape == arr2.shape)


    diff_abs = np.abs(arr2 - arr1)

    if epsilon is not None:
        diff_abs[diff_abs < epsilon] = 0

    diff_abs_rel = np.abs(diff_abs/arr1)

    print(f"Shape: {arr1.shape}")
    print(f"Média original: {np.mean(arr1)}")
    print(f"Média interpolada: {np.mean(arr2)}")
    print(f"Média da diferença: Absoluta = {np.mean(diff_abs)}, Relativa = {100*np.mean(diff_abs_rel)}%")

    max_diff_rel = np.max(diff_abs_rel)
    arg_max_diff_rel = np.argmax(diff_abs_rel)

    print(f"Maior diferença relativa: {100*max_diff_rel}%, em ({arg_max_diff_rel // diff_abs_rel.shape[1]}, {arg_max_diff_rel % diff_abs_rel.shape[1]})")

    print(f"Visualização em [{range_i[0]}, {range_i[1]}) x [{range_j[0]}, {range_j[1]})\n")

    print("Original")
    print_arr(arr1[range_i[0]:range_i[1], range_j[0]:range_j[1]])
    
    print("Interpolado")
    print_arr(arr2[range_i[0]:range_i[1], range_j[0]:range_j[1]])
    
    print("Diferença absoluta")
    print_arr(diff_abs[range_i[0]:range_i[1], range_j[0]:range_j[1]], round_digs=9)
    
    print("Diferença relativa (porcentagem)")
    print_arr(100 * diff_abs_rel[range_i[0]:range_i[1], range_j[0]:range_j[1]])


def interpol_array_from_csv(path, dec_row, dec_col, range_i=[30,41], range_j=[10,19]):
    
    
    arr_orig = pd.read_csv(path).to_numpy(dtype=np.double)
    arr_orig, arr_interp = interpolation(arr_orig, dec_row, dec_col)


    diffabsextrap = np.abs(arr_orig - arr_interp)
    diff = diffabsextrap[1::2,1::2]

    print(f"Shape original: {arr_orig.shape}")
    print(f"Shape da interpolação: {diff.shape}")
    print()

    print(f"Média dos pontos interpolados: {np.mean(arr_interp[1::2,1::2])}")
    print(f"Média do original nos pontos interpolados: {np.mean(arr_orig[1::2,1::2])}")
    print(f"Média da diferença: {np.mean(diff)}")
    print(f"Raíz da média quadrática da diferença: {np.sqrt(np.mean(np.square(diff)))}")
    print()
    print(f"Maior diferença absoluta: {np.max(diffabsextrap)}, em ({np.argmax(diffabsextrap)//diffabsextrap.shape[1]}, {np.argmax(diffabsextrap)%diffabsextrap.shape[1]})")
    relatdiffabsextrap = diffabsextrap/arr_orig
    print(f"Maior diferença percentual: {np.max(relatdiffabsextrap)}, em ({np.argmax(relatdiffabsextrap)//relatdiffabsextrap.shape[1]}, {np.argmax(relatdiffabsextrap)%relatdiffabsextrap.shape[1]})")

    #range_i = [30, 41]
    #range_j = [8, 19]

    #range_i = [92, 105]
    #range_j = [72, 85]


    print(f"\n\nVisualização em [{range_i[0]}, {range_i[1]}) x [{range_j[0]}, {range_j[1]})\n")

    print("Original")
    print_arr_interpol(arr_orig[range_i[0]:range_i[1], range_j[0]:range_j[1]])
    
    print("Interpolado")
    print_arr_interpol(arr_interp[range_i[0]:range_i[1], range_j[0]:range_j[1]])
    
    print("Diferença absoluta")
    print_arr_interpol(diffabsextrap[range_i[0]:range_i[1], range_j[0]:range_j[1]])
    
    print("Diferença relativa")
    print_arr_interpol(relatdiffabsextrap[range_i[0]:range_i[1], range_j[0]:range_j[1]])

def interpolation(arr_orig, dec_row, dec_col, method="linear"):
    dec_row = 2 # Adicionar outras interpolações
    dec_col = 2

    #print(arr_orig)
    # Tira as pontas do array, adequando o tamanho para que o primeiro e o último elemento estejam na versão decimada:
    max_row_slice = ((arr_orig.shape[0] - 1) // dec_row) * dec_row + 1
    max_col_slice = ((arr_orig.shape[1] - 1) // dec_col) * dec_col + 1
    arr_trimmed = arr_orig[:max_row_slice, :max_col_slice]

    #print("trimmmed")
    #print(arr_trimmed, end="\n\n")

    arr_dec = -1*np.ones(arr_trimmed.shape, dtype=np.double)
    arr_dec[::dec_row, ::dec_col] = arr_trimmed[::dec_row, ::dec_col]

    #print(f"dec, ({dec_row}, {dec_col}) ({max_row_slice}, {max_col_slice})")
    #print(arr_dec, end="\n\n")

    # Interpola ao longo das linhas
    for i in range(1, max_row_slice, 2):
        for j in range(0, max_col_slice, 2):
            if method == "linear":
                arr_dec[i, j] = (arr_dec[i - 1, j] + arr_dec[i + 1, j])/2

    #print("Após interpolação no eixo 0")
    #print(arr_dec, end="\n\n")

    # Interpola ao longo das colunas
    for j in range(1, max_col_slice, 2):
        for i in range(max_row_slice):
            if method == "linear":
                arr_dec[i, j] = (arr_dec[i, j - 1] + arr_dec[i, j + 1])/2

    #print("Após interpolação no eixo 1")
    #print(arr_dec, end="\n\n")

    return arr_trimmed, arr_dec


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
    #arr = create_3d_arr()

    arr = np.arange(3*7*6).reshape(3, 7, 6).astype(np.double) + 1
    print("p = 0:") 
    print_arr(arr[0])
    print("p = 1:") 
    print_arr(arr[1])
    print("p = 2:") 
    print_arr(arr[2])


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

    elif method == "lsm_interpol":
        result_interpol = local_sparsity_baseline_interpolation_wrapper(arr, freq_width_energy=3, freq_width_sparsity=3, time_width=3, zeta=10)


    elif method == "fls":
        print("\n\n\n==========\nCython:\n\n=============\n\n\n")
        result_cython = fast_local_sparsity_wrapper(arr, freq_width=3, time_width=3, eta=2)   
        print_arr(result_cython, round_digs=6)

        print("\n\n\n==========\nFeulo:\n\n=============\n\n\n")
        result_feulo = feulo_fast_local_sparsity_wrapper(arr, freq_width=3, time_width=3, eta=2)   
        print_arr(result_feulo, round_digs=6)

    elif method == "swgm":
        print("\n\n\n==========\nCython:\n\n=============\n\n\n")
        result_cython = swgm_wrapper(arr, beta=0.3, max_gamma=20.0)   
        print_arr(result_cython, round_digs=6)

        print("\n\n\n==========\nFeulo:\n\n=============\n\n\n")
        result_feulo = feulo_swgm_wrapper(arr, beta=0.3, max_gamma=20.0)   
        print_arr(result_feulo, round_digs=6)

    else:
        print("Especifique o método")


if __name__ == "__main__":
    #test_method("lt")
    #test_method("lsm")
    #test_method("lsm_interpol")
    test_method("swgm")

    #compare_representations("backup/normal.bkp", "backup/interpol.bkp", range_i=[1820, 1833], range_j=[24, 35]) # Região de baixa energia, altos erros sem o epsilon.
    #compare_representations("backup/normal.bkp", "backup/interpol.bkp", range_i = [48, 65], range_j = [0, 17]) # Região de alta energia
    #compare_representations("backup/normal.bkp", "backup/interpol.bkp", range_i = [76, 89], range_j = [92, 105]) # Região de maior erro relativo com o epsilon, transição entre energias.


    #interpol_array_from_csv("dev/csv_smearing/2048.csv", 2, 2, range_i = [30, 41], range_j = [10, 19])
    #interpol_array_from_csv("dev/csv_sparsity/2048.csv", 2, 2, range_i = [92, 105], range_j = [72, 85])