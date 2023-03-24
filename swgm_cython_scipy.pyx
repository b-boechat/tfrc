import numpy as np
from scipy.stats import gmean
cimport cython
from libc.math cimport exp, log 

def swgm_cython_scipy_wrapper(X, beta=0.3, max_gamma=20.0):
    return swgm_cython_scipy(X, beta, max_gamma)

cdef swgm_baseline_scipy(double[:,:,::1] X, double beta, double max_gamma):
    cdef:
        Py_ssize_t P = X.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X.shape[1] # Eixo das frequências
        Py_ssize_t M = X.shape[2] # Eixo do tempo

        Py_ssize_t p, k, m aux_p
        double epsilon = 1e-10
    
    gammas_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] gammas = gammas_ndarray
    # =====
    for k in range(K):
        for m in range(M):
            for p in range(P):
                gammas[p, k, m] = -log(X[p, k, m] + epsilon)
                for aux_p in range(P):
                    gammas[p, k, m] += log(X[aux_p, k, m] + epsilon)
                gammas[p, k, m] = exp(gammas[p, k, m] * beta)
                if gammas[p, k, m] > max_gamma:
                    gammas[p, k, m] = max_gamma
    # =====
    return gmean(X, axis=0, weights=gammas_ndarray)
    
