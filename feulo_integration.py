from noname import spectrogram_local_sparsity, spectrogram_comb_Lukin_Todd, spectrogram_comb_FastHoyerLocalSparsity, swgm_comb
import numpy as np

def feulo_spectrogram_local_sparsity_wrapper(X, freq_width_sparsity=39, freq_width_energy=15, time_width=11, zeta=80):
    return spectrogram_local_sparsity(np.transpose(X, (1, 2, 0)), 
            size_w_S=[freq_width_sparsity, time_width], 
            size_w_E=[freq_width_energy, time_width],
            zeta=zeta, eta=1)

def feulo_lukin_todd_wrapper(X, freq_width=39, time_width=11, eta=8.0):
    return spectrogram_comb_Lukin_Todd(np.transpose(X, (1, 2, 0)), size_w_S=[freq_width, time_width], eta=eta)

def feulo_fast_local_sparsity_wrapper(X, freq_width=39, time_width=11, eta=8.0):
    return spectrogram_comb_FastHoyerLocalSparsity(np.transpose(X, (1, 2, 0)), size_W_m_k=[freq_width, time_width], eta=eta)

def feulo_swgm_wrapper(X, beta=0.3, max_gamma=20):
    return swgm_comb(np.transpose(X, (1, 2, 0)), beta=beta, max_alpha=max_gamma)
