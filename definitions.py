#import pyximport; pyximport.install(language_level="3")


#def func_not_installed(X, **kwargs): # TODO Criar uma exception para FunctionNotInstalled posteriormente.
#    raise Exception

from binwise_combination import median_combination, mean_combination
from feulo_integration import feulo_spectrogram_local_sparsity_wrapper, \
                            feulo_lukin_todd_wrapper, \
                            feulo_fast_local_sparsity_wrapper, \
                            feulo_swgm_wrapper
from matlab_integration import fast_local_sparsity_matlab_wrapper, \
                            local_sparsity_matlab_wrapper, \
                            fast_local_sparsity_matlab_wrapper, \
                            lukin_todd_matlab_wrapper
                            #sample_weighted_geometric_mean_matlab_wrapper


#import subprocess
#subprocess.call("python setup.py build_ext --inplace")  # TODO solução temporária até encontrar uma forma melhor de instalar o módulo em Cython por script.

from lsm import local_sparsity_wrapper
from lsm_baseline import local_sparsity_baseline_wrapper

from lsm_baseline_interpol import local_sparsity_baseline_interpolation_wrapper
from lsm_interpol_v1 import local_sparsity_interpolation_v1_wrapper

from lsm_hybrid import local_sparsity_hybrid_wrapper

from lukin_todd import lukin_todd_wrapper
from lukin_todd_v1 import lukin_todd_v1_wrapper
from lukin_todd_baseline import lukin_todd_baseline_wrapper

from fls import fast_local_sparsity_wrapper
from fls_hybrid import fast_local_sparsity_hybrid_wrapper
from fls_hybrid_bin import fast_local_sparsity_hybrid_bin_wrapper

from swgm import swgm_cython_scipy_wrapper, swgm_cython_scipy_presum_wrapper
from swgm_scipy import swgm_scipy_v1, swgm_scipy_v2, swgm_scipy_v3, swgm_scipy_v4


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
    
    "lsm_interpol_v1": {
        "name" : "Local Sparsity Method with Interpolation (v1)",
        "function" : local_sparsity_interpolation_v1_wrapper
    }, 

    "lsm_baseline_interpol": {
        "name" : "Local Sparsity Method with Interpolation (Baseline)",
        "function" : local_sparsity_baseline_interpolation_wrapper
    }, 
    "lsm_hybrid": {
        "name" : "Local Sparsity Method (Hybrid)",
        "function" : local_sparsity_hybrid_wrapper
    }, 
    "lsm_feulo": {
        "name" : "Local Sparsity (Feulo)",
        "function" : feulo_spectrogram_local_sparsity_wrapper
    },
    "lsm_matlab": {
        "name" : "Local Sparsity (Maurício)",
        "function" : local_sparsity_matlab_wrapper
    },
    "fls": {
        "name" : "Fast Local Sparsity",
        "function" : fast_local_sparsity_wrapper
    },
    "fls_hybrid": {
        "name" : "Fast Local Sparsity (Hybrid)",
        "function" : fast_local_sparsity_hybrid_wrapper
    },
    "fls_hybrid_bin": {
        "name" : "Fast Local Sparsity (Hybrid with binwise criterium)",
        "function" : fast_local_sparsity_hybrid_bin_wrapper
    },
    "fls_feulo": {
        "name" : "Fast Local Sparsity (Feulo)",
        "function" : feulo_fast_local_sparsity_wrapper
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
    "swgm_cython_scipy": {
        "name" : "Sample Weighted Geometric Mean",
        "function" : swgm_cython_scipy_wrapper
    },
    "swgm_cython_scipy_presum": {
        "name" : "Sample Weighted Geometric Mean (Presum)",
        "function" : swgm_cython_scipy_presum_wrapper
    },
    "swgm_feulo": {
        "name" : "Sample Weighted Geometric Mean (Feulo)",
        "function" : feulo_swgm_wrapper
    },
    "swgm_scipy_v1": {
        "name" : "Sample Weighted Geometric Mean (Scipy v1)",
        "function" : swgm_scipy_v1
    },
    "swgm_scipy_v2": {
        "name" : "Sample Weighted Geometric Mean (Scipy v2)",
        "function" : swgm_scipy_v2
    },
    "swgm_scipy_v3": {
        "name" : "Sample Weighted Geometric Mean (Scipy v3)",
        "function" : swgm_scipy_v3
    },
    "swgm_scipy_v4": {
        "name" : "Sample Weighted Geometric Mean (Scipy v4)",
        "function" : swgm_scipy_v4
    },

    #"swgm_matlab": {
    #    "name" : "Sample Weighted Geometric Mean (Maurício)",
    #    "function" : sample_weighted_geometric_mean_matlab_wrapper
    #}
}