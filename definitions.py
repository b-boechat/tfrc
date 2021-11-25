from binwise_combination import median_combination, mean_combination
from local_sparsity import local_sparsity_method

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
        "function" : local_sparsity_method
    }
}

backup_files_extension = ".bkp"
audio_folder = "audio"