import numpy as np
from scipy.stats import gmean

def swgm_scipy_v1(X, beta=0.3, max_gamma=20.0):
    P, K, M = X.shape
    gammas = np.zeros((P, K, M), dtype=np.double)
    # ====
    for p in range(P):
        gammas[p, :, :] = (gmean(np.delete(X, p, axis=0), axis=0)/X[p, :, :]) ** beta
    # ====
    gammas[np.bitwise_or(gammas > max_gamma, np.isnan(gammas))] = max_gamma 
    return gmean(X, axis=0, weights=gammas)


def swgm_scipy_v2(X, beta=0.3, max_gamma=20.0):
    P, K, M = X.shape
    gammas = np.zeros((P, K, M), dtype=np.double)
    
    # =====
    product_holder = np.prod(X, axis=0)
    for p in range(P):
        gammas[p, :, :] = (np.power(product_holder / X[p, :, :], 1/(P - 1))/X[p, :, :]) ** beta
    # =====
    gammas[np.bitwise_or(gammas > max_gamma, np.isnan(gammas))] = max_gamma 
    return gmean(X, axis=0, weights=gammas)

def swgm_scipy_v3(X, beta=0.3, max_gamma=20.0):
    P, K, M = X.shape
    gammas = np.zeros((P, K, M), dtype=np.double)
    
    # =====
    product_power_holder = np.exp(np.sum(np.log(X), axis=0) / P - 1)
    for p in range(P):
        gammas[p, :, :] = (product_power_holder / np.power(X[p, :, :], P/(P - 1))) ** beta
    # =====
    gammas[np.bitwise_or(gammas > max_gamma, np.isnan(gammas))] = max_gamma 
    return gmean(X, axis=0, weights=gammas)

def swgm_scipy_v4(X, beta=0.3, max_gamma=20.0):
    P, K, M = X.shape
    gammas = np.zeros((P, K, M), dtype=np.double)
    
    # =====
    sum_ln_holder = np.sum(np.log(X), axis=0) / P - 1
    for p in range(P):
        gammas[p, :, :] = np.exp((sum_ln_holder - np.log(X[p, :, :])*P/(P - 1) ) * beta)
    # =====
    gammas[np.bitwise_or(gammas > max_gamma, np.isnan(gammas))] = max_gamma 
    
    return gmean(X, axis=0, weights=gammas)