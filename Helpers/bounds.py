import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
import torch

# t< R R==alpha/pred; t== 
def f(t, R):
    t = np.asarray(t)  # Ensure t is an array (even if it is a single value)
    
    result = np.zeros_like(t, dtype=float)  # Initialize a result array
    
    # Case for t == 0
    mask_0 = (t == 0)
    result[mask_0] = (1 - t[mask_0]) * np.log((1 - t[mask_0]) / (1 - R)) * (t[mask_0] < R)
    
    # Case for t == 1
    mask_1 = (t == 1)
    result[mask_1] = t[mask_1] * np.log(t[mask_1] / R) * (t[mask_1] <= R)
    
    # Default case
    mask_default = ~(mask_0 | mask_1)
    result[mask_default] = (t[mask_default] * np.log(t[mask_default] / R) + 
                             (1 - t[mask_default]) * np.log((1 - t[mask_default]) / (1 - R))) * (t[mask_default] < R)
    
    return result

def f_naive(t, R):
    return 2 * (np.maximum(R - t, 0)**2)

def naive_hoeffding_inequality_function(t, R, n):
    return -n * f_naive(t,R)

def hoeffding_inequality_function(t, R, n):
    return -n * f(t,R)

def bentkus_inequality_function(t, R, n):
    cdf_values = binom.cdf(np.ceil(n * t), n, R)
    cdf_values = np.maximum(cdf_values, 1e-10)  # Avoid log of zero
    return np.log(cdf_values) + 1

def HB_UCB(Rhat, n, delta, maxiters = int(1e5)):
    #Proposition 1.
    def _tailprob(R):
        hoeffding_R = hoeffding_inequality_function(Rhat, R, n) 
        bentkus_R = bentkus_inequality_function(Rhat, R, n)
        return min(hoeffding_R, bentkus_R) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, Rhat, 1-1e-10, maxiter=maxiters)
    
def hoeffding_UCB(Rhat, n, delta, maxiters = int(1e5)):
    #Proposition 1.
    def _tailprob(R):
        return hoeffding_inequality_function(Rhat, R, n) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, Rhat, 1-1e-10, maxiter=maxiters)
    
def bentkus_UCB(Rhat, n, delta, maxiters = int(1e5)):
    #Proposition 1.
    def _tailprob(R):
        return bentkus_inequality_function(Rhat, R, n) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, Rhat, 1-1e-10, maxiter=maxiters)