#import pyximport; pyximport.install(language_level="3")


#def func_not_installed(X, **kwargs): # TODO Criar uma exception para FunctionNotInstalled posteriormente.
#    raise Exception

import pyximport
#pyximport.install(language_level="3")

from lsm import local_sparsity_wrapper
from lsm2 import local_sparsity_before_par_wrapper

from binwise_combination import median_combination, mean_combination
from local_sparsity import local_sparsity_method_python


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
    "lsmold": {
        "name" : "Local Sparsity Method (Old)",
        "function" : local_sparsity_method_python
    },
    "lsmbeforepar": {
        "name" : "Local Sparsity Method (before changes for parallelization)",
        "function" : local_sparsity_before_par_wrapper
    }
}