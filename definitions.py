import pyximport; pyximport.install(language_level="3")

from lsm import local_sparsity_wrapper
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
    }
}