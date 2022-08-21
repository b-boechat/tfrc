# Autor: Guilherme Feulo

import numpy as np
from scipy.signal import correlate

def spectrogram_comb_FastHoyerLocalSparsity(spectrograms_tensor, size_W_m_k, eta):
    epsilon = 1e-10
    Hoyer_local_sparsity_measure_tensor = np.zeros_like(spectrograms_tensor)
    LocalSparsity_ratio_all = Hoyer_local_sparsity_measure_tensor.copy()
    if size_W_m_k[0] % 2 == 0:    
        size_W_m_k[0] += 1
        print(f'WARNING: size_w_S(1) must be an odd number! Using {size_W_m_k[0]} instead!')

    if size_W_m_k[1] % 2 == 0:
        size_W_m_k[1] += 1
        print(f'WARNING: size_w_S(2) must be an odd number! Using {size_W_m_k[1]} instead!')
        
    wr = np.hamming(size_W_m_k[0])
    wc = np.hamming(size_W_m_k[1])
    maskr, maskc = np.meshgrid(wc, wr)
    window_S = maskc * maskr
    
    N = size_W_m_k[0] * size_W_m_k[1]
    for spec_ind in range(spectrograms_tensor.shape[-1]):

        local_energy = correlate(spectrograms_tensor[:,:,spec_ind], window_S, mode='same') + epsilon
        local_energy_2 = correlate(spectrograms_tensor[:,:,spec_ind] ** 2, window_S ** 2, mode='same') + epsilon
       
        Hoyer_local_sparsity_measure_tensor[:, :, spec_ind] = (np.sqrt(N) - local_energy / np.sqrt(local_energy_2)) / ((np.sqrt(N) - 1) * np.sqrt(local_energy))
    
    for spec_ind in range(spectrograms_tensor.shape[2]):
        LocalSparsity_spects_aux = Hoyer_local_sparsity_measure_tensor + epsilon
        LocalSparsity_spects_aux = np.delete(LocalSparsity_spects_aux, spec_ind, axis=2)
        LocalSparsity_ratio_all[:,:,spec_ind] = np.power(Hoyer_local_sparsity_measure_tensor[:,:,spec_ind] / np.prod(LocalSparsity_spects_aux, axis=2), eta)
        LocalSparsity_ratio_all[np.isnan(LocalSparsity_ratio_all[:,:,spec_ind])] = np.max(LocalSparsity_ratio_all[:,:,spec_ind])
    
    final_TFR = np.sum(spectrograms_tensor * LocalSparsity_ratio_all, axis=2) / np.sum(LocalSparsity_ratio_all, axis=2)
    final_TFR[np.isnan(final_TFR)] = 0

    orig_energy = np.sum(spectrograms_tensor[:,:,0])
    comb_energy = np.sum(final_TFR)

    return final_TFR * orig_energy / comb_energy