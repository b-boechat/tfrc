import numpy as np
from math import ceil

def computeGiniIndex(S):
    
    if len(S.shape) == 2:
        S = S.reshape(*S.shape, 1)

    sparsity = np.zeros(S.shape[2])
    for tfrInd in range(S.shape[2]):
        if np.sum(S[:,:,tfrInd]) > 0:
            
            vector = S.flatten('F')
            
            sortedVector = np.sort(vector)
            sortedVector = sortedVector/np.sum(sortedVector)

            N = len(sortedVector)
            k = np.arange(1, N+1)
            auxVector = (N - k + 0.5)/N

            sparsity[tfrInd] = 1 - 2*np.sum(sortedVector * auxVector)
        else:
            sparsity[tfrInd] = 0

    return sparsity

def spectrogram_local_sparsity(spectrograms_tensor, size_w_S, size_w_E, zeta, eta=1):
    epslon = 1e-10
    size_w_S = list(map(ceil, size_w_S))
    size_w_E = list(map(ceil, size_w_E))

    if size_w_S[0] % 2 == 0:    
        size_w_S[0] += 1
        print(f'WARNING: size_w_S(1) must be an odd number! Using {size_w_S[0]} instead!')

    if size_w_S[1] % 2 == 0:    
        size_w_S[1] += 1
        print(f'WARNING: size_w_S(1) must be an odd number! Using {size_w_S[1]} instead!')

    if size_w_E[0] % 2 == 0:    
        size_w_E[0] += 1
        print(f'WARNING: size_w_S(1) must be an odd number! Using {size_w_E[0]} instead!')

    if size_w_E[1] % 2 == 0:    
        size_w_E[1] += 1
        print(f'WARNING: size_w_S(1) must be an odd number! Using {size_w_E[1]} instead!')
    
    LocalSparsity_ratio_all = np.zeros_like(spectrograms_tensor)
    LocalSparsity = np.zeros_like(spectrograms_tensor)
    LocalEnergy = np.zeros_like(spectrograms_tensor)

    wr = np.hamming(size_w_S[0])
    wc = np.hamming(size_w_S[1])
    maskr, maskc = np.meshgrid(wc, wr)
    window_S = maskc * maskr

    wr = np.hamming(size_w_E[0])
    wc = np.hamming(size_w_E[1])
    wc[len(wc)//2+1:] = 0
    maskr, maskc = np.meshgrid(wc, wr)
    window_E  = maskc * maskr

    max_size_w = np.max([size_w_E, size_w_S], axis=0)
    zp_specs = np.pad(spectrograms_tensor, [(max_size_w[0]//2, max_size_w[0]//2), (max_size_w[1]//2, max_size_w[1]//2), (0,0)])

    Sparsity_vec_aux =  np.zeros(spectrograms_tensor.shape[2])
    energy_frames = np.sum(spectrograms_tensor[:,:,-1], axis=0)

    for col in range(max_size_w[1]//2, spectrograms_tensor.shape[1] + max_size_w[1]//2):
        if energy_frames[col - max_size_w[1]//2] > 0:
            for row in range(max_size_w[0]//2, spectrograms_tensor.shape[0] +max_size_w[0]//2):
                for spec_ind in range(spectrograms_tensor.shape[2]):
                    regionMatrixS = zp_specs[row-size_w_S[0]//2:row+size_w_S[0]//2+1, col-size_w_S[1]//2:col+size_w_S[1]//2+1,spec_ind]
                    windowedRegionSparsity = regionMatrixS * window_S
                    regionMatrixE = zp_specs[row-size_w_E[0]//2:row+size_w_E[0]//2+1, col-size_w_E[1]//2:col+size_w_E[1]//2+1,spec_ind]
                    windowedRegionEnergy = regionMatrixE * window_E
                    
                    # Computing local sparsity and energy
                    Sparsity = computeGiniIndex(windowedRegionSparsity)
                    LocalSparsity[row - max_size_w[0]//2, col - max_size_w[1]//2, spec_ind] = Sparsity
                    Sparsity_vec_aux[spec_ind] = Sparsity
                    LocalEnergy[row - max_size_w[0]//2, col - max_size_w[1]//2, spec_ind] = np.sum(windowedRegionEnergy)

    LocalEnergy[LocalEnergy < epslon] = LocalEnergy[LocalEnergy < epslon] + epslon # + ou = To avoid 0
    LocalSparsity[LocalSparsity < epslon] = LocalSparsity[LocalSparsity < epslon] + epslon # To avoid 0
    LocalEnergy_spects_min = np.min(LocalEnergy, axis=2)

    for spec_ind in range(spectrograms_tensor.shape[2]):
        LocalSparsity_spects_aux = np.delete(LocalSparsity, spec_ind, axis=2)
        LocalSparsity_ratio_all[:,:,spec_ind] = (LocalSparsity[:,:,spec_ind]/np.prod(LocalSparsity_spects_aux, axis=2)) ** zeta
        LocalSparsity_ratio_all[np.isnan(LocalSparsity_ratio_all[:,:,spec_ind])] = np.max(LocalSparsity_ratio_all[:,:,spec_ind])
    
    spects_matrix = spectrograms_tensor.copy()
    for spec_ind in range(spectrograms_tensor.shape[2]):
        # Energy normalization
        LocalEnergy_ratio = LocalEnergy_spects_min /LocalEnergy[:,:,spec_ind]
        spects_matrix[:,:,spec_ind] = spects_matrix[:,:,spec_ind] * LocalEnergy_ratio ** eta

    final_TFR = np.sum(spects_matrix * LocalSparsity_ratio_all,axis=2) / np.sum(LocalSparsity_ratio_all,axis=2)
    final_TFR[np.isnan(final_TFR)] = 0
    # energia é assim ou vai virar a do sinal ?
    orig_energy = np.sum(spects_matrix[:,:,0])
    comb_energy = np.sum(final_TFR)
    #return (orig_energy/comb_energy) * final_TFR
    return final_TFR
