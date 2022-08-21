# Autor: Guilherme Feulo

from re import I
import numpy as np

def compute_smearing(S):
    try:
        energyCompaction = np.zeros(S.shape[2])
    except IndexError:
        energyCompaction = np.zeros(1)
    
    for tfrInd in range(len(energyCompaction)):
        vector = S.flatten('F')
        sortedVector = -np.sort(-vector)
        N = len(sortedVector)
        indVector = np.arange(1, N+1)
        firstMoment = np.sum(sortedVector * indVector)
        energyCompaction[tfrInd] = firstMoment/(np.sqrt(np.sum(sortedVector)) + 1e-10)
    return energyCompaction

def spectrogram_comb_Lukin_Todd(spectrograms_tensor, size_w_S, eta):
    epsilon = 1e-10  # avoid division by 0
    #size_w_S = ceil(size_w_S);

    energy_save = 0 # This is used to save processing when there is no energy in a given frame

    if size_w_S[0] % 2 == 0:    
        size_w_S[0] += 1
        print(f'WARNING: size_w_S(1) must be an odd number! Using {size_w_S[0]} instead!')


    if size_w_S[1] % 2 == 0:
        size_w_S[1] += 1
        print(f'WARNING: size_w_S(2) must be an odd number! Using {size_w_S[1]} instead!')

    # Zero-padding the spectrograms to properly apply the 2D windows at the edges
    max_size_w = size_w_S
    
    zp_specs = np.pad(spectrograms_tensor, [(max_size_w[0]//2, max_size_w[0]//2), (max_size_w[1]//2, max_size_w[1]//2), (0,0)])
    
    localSmearing_tensor =  np.zeros_like(spectrograms_tensor)
    energy_frames = np.sum(spectrograms_tensor[:,:,-1], axis=0)
    for col in range(max_size_w[1]//2, spectrograms_tensor.shape[1] + max_size_w[1]//2):
        if not energy_save or energy_frames[col - max_size_w[1]//2] > 0:
            for row in range(max_size_w[0]//2, spectrograms_tensor.shape[0] +max_size_w[0]//2):
                for spec_ind in range(spectrograms_tensor.shape[2]):
                    regionMatrix = zp_specs[row-size_w_S[0]//2:row+size_w_S[0]//2+1, col-size_w_S[1]//2:col+size_w_S[1]//2+1,spec_ind]
                    windowedRegion = regionMatrix
                    localSmearing = compute_smearing(windowedRegion)
                    localSmearing_tensor[row-max_size_w[0]//2, col-max_size_w[1]//2, spec_ind] = localSmearing

    #for spec_ind in range(spectrograms_tensor.shape[2]):
    #    print(f"p={spec_ind}")
    #    print(localSmearing_tensor[50:65,50:65,spec_ind])

    weight = 1/(localSmearing_tensor ** eta + epsilon)
    final_TFR = np.sum(spectrograms_tensor * weight, axis=2) / np.sum(weight, axis=2)
    #orig_energy = np.sum(spectrograms_tensor[:,:,0])
    #comb_energy = np.sum(final_TFR)
    #return final_TFR*orig_energy/comb_energy
    return final_TFR

                