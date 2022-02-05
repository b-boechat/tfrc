from noname import spectrogram_local_sparsity
import numpy as np

def feulo_spectrogram_local_sparsity_wrapper(X, freq_width_sparsity=17, freq_width_energy=41, time_width=13, zeta=80):
    return spectrogram_local_sparsity(np.transpose(X, (1, 2, 0)), 
            size_w_S=[freq_width_sparsity, time_width], 
            size_w_E=[freq_width_energy, time_width],
            zeta=zeta, eta=1)
