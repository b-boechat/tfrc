#import pyximport; pyximport.install(language_level="3")


#def func_not_installed(X, **kwargs): # TODO Criar uma exception para FunctionNotInstalled posteriormente.
#    raise Exception

from binwise_combination import median_combination, mean_combination
from feulo_integration import feulo_spectrogram_local_sparsity_wrapper, \
                            feulo_lukin_todd_wrapper
from matlab_integration import fast_local_sparsity_matlab_wrapper, \
                            local_sparsity_matlab_wrapper, \
                            fast_local_sparsity_matlab_wrapper, \
                            lukin_todd_matlab_wrapper, \
                            sample_weighted_geometric_mean_matlab_wrapper


#import subprocess
#subprocess.call("python setup.py build_ext --inplace")  # TODO solução temporária até encontrar uma forma melhor de instalar o módulo em Cython por script.

from lsm import local_sparsity_wrapper
from lsm_baseline import local_sparsity_baseline_wrapper
from lukin_todd import lukin_todd_wrapper
from lukin_todd_v1 import lukin_todd_v1_wrapper
from lukin_todd_baseline import lukin_todd_baseline_wrapper
from swgm import swgm_wrapper



backup_files_extension = ".bkp"
backup_folder = "backup"
audio_folder = "audio"


combination_methods = {
    "median": {
        "name" : "Binwise Median",
        "function" : median_combination
    },
    "mean": {
        "name" : "Binwise Mean",
        "function" : mean_combination
    },
    "lsm": {
        "name" : "Local Sparsity Method",
        "function" : local_sparsity_wrapper
    },
    "lsm_baseline": {
        "name" : "Local Sparsity Method (Baseline)",
        "function" : local_sparsity_baseline_wrapper
    },
    "lsm_feulo": {
        "name" : "Local Sparsity (Feulo)",
        "function" : feulo_spectrogram_local_sparsity_wrapper
    },
    "lsm_matlab": {
        "name" : "Local Sparsity (Maurício)",
        "function" : local_sparsity_matlab_wrapper
    },
    "fls_matlab": {
        "name" : "Fast Local Sparsity (Maurício)",
        "function" : fast_local_sparsity_matlab_wrapper
    },
    "lt": {
        "name" : "Lukin Todd",
        "function" : lukin_todd_wrapper 
    },
    "lt_v1": {
        "name" : "Lukin Todd (v1)",
        "function" : lukin_todd_v1_wrapper 
    },
    "lt_feulo": {
        "name" : "Lukin Todd (Feulo)",
        "function" : feulo_lukin_todd_wrapper 
    },
    "lt_baseline": {
        "name" : "Lukin Todd (Baseline)",
        "function" : lukin_todd_baseline_wrapper
    },
    "lt_matlab": {
        "name" : "Lukin Todd (Maurício)",
        "function" : lukin_todd_matlab_wrapper 
    },
    "swgm": {
        "name" : "Sample Weighted Geometric Mean",
        "function" : swgm_wrapper
    },
    "swgm_matlab": {
        "name" : "Sample Weighted Geometric Mean (Maurício)",
        "function" : sample_weighted_geometric_mean_matlab_wrapper
    }
}