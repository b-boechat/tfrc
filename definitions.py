#import pyximport; pyximport.install(language_level="3")


#def func_not_installed(X, **kwargs): # TODO Criar uma exception para FunctionNotInstalled posteriormente.
#    raise Exception

import subprocess
from binwise_combination import median_combination, mean_combination
from local_sparsity import local_sparsity_method_python




subprocess.call("python setup.py build_ext --inplace")  # TODO solução temporária até encontrar uma forma melhor de instalar o módulo em Cython por script.
from lsm_par import local_sparsity_parallel_wrapper
from lsm import local_sparsity_wrapper




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
    "lsm_par": {
        "name" : "Local Sparsity Method (parallel)",
        "function" : local_sparsity_parallel_wrapper
    },
    "lsm_pure_python": {
        "name" : "Local Sparsity (Pure Python)",
        "function" : local_sparsity_method_python
    }
    
}