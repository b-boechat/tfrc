from noname import spectrogram_local_sparsity, spectrogram_comb_Lukin_Todd
import numpy as np

def feulo_spectrogram_local_sparsity_wrapper(X, freq_width_sparsity=17, freq_width_energy=41, time_width=13, zeta=80):
    return spectrogram_local_sparsity(np.transpose(X, (1, 2, 0)), 
            size_w_S=[freq_width_sparsity, time_width], 
            size_w_E=[freq_width_energy, time_width],
            zeta=zeta, eta=1)

def feulo_lukin_todd_wrapper(X, freq_width=17, time_width=13, eta=8):
    return spectrogram_comb_Lukin_Todd(np.transpose(X, (1, 2, 0)), size_w_S=[freq_width, time_width], eta=eta)