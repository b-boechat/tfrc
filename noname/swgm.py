# Autor: Guilherme Feulo

import numpy as np

def geo_mean(array, axis=None):
    return array.prod(axis=axis)**(1.0/array.shape[axis])

def swgm(spectrograms_tensor, beta, max_alpha):
    alphas = np.zeros_like(spectrograms_tensor)
    if beta == 0:
        spec_prod_w = geo_mean(spectrograms_tensor, 2)
    else:
        for spec_ind in range(spectrograms_tensor.shape[2]):
            spec_matrix_aux = np.delete(spectrograms_tensor, spec_ind, axis=2)
            spects_geomean = geo_mean(spec_matrix_aux, 2)

            alphas[:,:,spec_ind] = (spects_geomean / spectrograms_tensor[:,:,spec_ind]) ** beta
            alphas[np.isnan(alphas[:,:,spec_ind])] = max_alpha
            alphas[alphas > max_alpha] = max_alpha

        spects_matrix_alphas  = spectrograms_tensor ** alphas
        spec_prod_w = spects_matrix_alphas.prod(axis=2)**(1.0/np.sum(alphas, axis=2))

    #orig_energy = np.sum(spectrograms_tensor[:,:,0])
    #comb_energy = np.sum(spec_prod_w)

    return spec_prod_w

