import numpy as np
from scipy.stats import gmean
cimport cython
from libc.math cimport pow, exp, log

def swgm_cython_scipy_wrapper(X, beta=0.3, max_gamma=20.0):
    return swgm_cython_scipy(X, beta, max_gamma)

def swgm_cython_scipy_presum_wrapper(X, beta=0.3, max_gamma=20.0):
    return swgm_cython_scipy_presum(X, beta, max_gamma)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef swgm_cython_scipy(double[:,:,::1] X, double beta, double max_gamma):
    cdef:
        Py_ssize_t P = X.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X.shape[1] # Eixo das frequências
        Py_ssize_t M = X.shape[2] # Eixo do tempo

        Py_ssize_t p, k, m, aux_p
        double epsilon = 1e-10

    log_X_ndarray = np.log(np.asarray(X), dtype=np.double)
    cdef double[:, :, :] log_X = log_X_ndarray
    
    gammas_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] gammas = gammas_ndarray
    # =====
    for k in range(K):
        for m in range(M):
            for p in range(P):
                gammas[p, k, m] = -log_X[p, k, m]
                for aux_p in range(P):
                        gammas[p, k, m] += log_X[aux_p, k, m]
                gammas[p, k, m] /= P - 1
                gammas[p, k, m] -= log_X[p, k, m]

                gammas[p, k, m] = exp(gammas[p, k, m] * beta)
                if gammas[p, k, m] > max_gamma:
                    gammas[p, k, m] = max_gamma
    
    # =====
    return gmean(X, axis=0, weights=gammas_ndarray)


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef swgm_cython_scipy_presum(double[:,:,::1] X, double beta, double max_gamma):
    cdef:
        Py_ssize_t P = X.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X.shape[1] # Eixo das frequências
        Py_ssize_t M = X.shape[2] # Eixo do tempo

        Py_ssize_t p, k, m, aux_p
        double epsilon = 1e-10

    log_X_ndarray = np.log(np.asarray(X), dtype=np.double)
    cdef double[:, :, :] log_X = log_X_ndarray
    
    gammas_ndarray = np.array([np.sum(log_X_ndarray, axis=0) / (P - 1)] * P)
    cdef double[:,:,:] gammas = gammas_ndarray
    
    # =====
    for k in range(K):
        for m in range(M):
            for p in range(P):
                gammas[p, k, m] -= log_X[p, k, m] * P / (P - 1)
                gammas[p, k, m] = exp(gammas[p, k, m] * beta)
                if gammas[p, k, m] > max_gamma:
                    gammas[p, k, m] = max_gamma
    
    # =====
    return gmean(X, axis=0, weights=gammas_ndarray)