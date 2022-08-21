import numpy as np
import matlab
import matlab.engine


def matlab_wrapper(path, func_name, *args):

    eng = matlab.engine.start_matlab()
    eng.addpath(path, nargout=0)

    assert hasattr(eng, func_name)
    func = getattr(eng, func_name)

    result = func(*args)

    result_ndarray = np.array(result._data).reshape(result.size, order='F')

    eng.quit()

    return result_ndarray



def local_sparsity_matlab_wrapper(X, freq_width_sparsity=39, freq_width_energy=15, time_width=11, zeta=80):
    return matlab_wrapper("TFR_Methods/SLS", "spectrogram_comb_local_sparsity", 
            matlab.double(np.transpose(X, (1, 2, 0)).tolist()), #specs_matrix
            matlab.double([freq_width_sparsity, time_width]), # size_W_S
            matlab.double([freq_width_energy, time_width]), # size_W_E
            float(zeta), #zeta 
            float(1)) #eta
            

def fast_local_sparsity_matlab_wrapper(X, freq_width=39, time_width=11, eta=8):
    return matlab_wrapper("TFR_Methods/FHLS", "spectrogram_comb_FastHoyerLocalSparsity",
        matlab.double(np.transpose(X, (1, 2, 0)).tolist()),
        matlab.double([freq_width, time_width]),
        float(eta)
    )

def lukin_todd_matlab_wrapper(X, freq_width=39, time_width=11, eta=8):
    return matlab_wrapper("TFR_Methods/LT", "spectrogram_comb_Lukin_Todd", 
            matlab.double(np.transpose(X, (1, 2, 0)).tolist()), #specs_matrix
            matlab.double([freq_width, time_width]), # size_W_S
            float(eta), #eta 
    )

def sample_weighted_geometric_mean_matlab_wrapper(X, beta=0.5):
    return matlab_wrapper("TFR_Methods/SWGM", "SWGM_comb", 
            matlab.double(np.transpose(X, (1, 2, 0)).tolist()), #specs_matrix
            float(beta) #beta
    ) 