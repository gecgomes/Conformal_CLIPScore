from Helpers.plots import plot_risks_and_pvalues
from Helpers.ci import process_truncated_gaussian, process_truncated_gaussian_mean, process_truncated_gaussian_std
from Helpers.bounds import HB_UCB, bentkus_inequality_function, hoeffding_UCB, hoeffding_inequality_function, naive_hoeffding_inequality_function
from scipy.optimize import brentq
import torch
import numpy as np
import tqdm

def conformal_risk_control_using_concentration_inequalities(prediction_set,Y,risk_fn,alpha,delta,n,maxiters =int(1e8)):
    def _get_UCB(l):
        predictions = []
        for item in prediction_set:
            predictions.append(item > l)
        Rhat =  risk_fn(predictions, Y)
        return HB_UCB(Rhat, n, delta, maxiters) - alpha
    _lambda = brentq(_get_UCB, 0, 1, maxiter=maxiters)
    return _lambda


def conformal_risk_control_using_LTT(prediction_data, Y, risk_fn, lower_bound,upper_bound, alpha,delta, n, N, start_lambda,stop_lambda):
    lambdas = np.linspace(start_lambda,stop_lambda,N)
    risks = np.zeros(N)
    for i,l in enumerate(tqdm.tqdm(lambdas, desc="Calibrating...")):
        T = process_truncated_gaussian_mean(prediction_data,l,lower_bound,upper_bound)
        S = process_truncated_gaussian_std(prediction_data,l,lower_bound,upper_bound)
        risks[i] = risk_fn(T,S,Y)
    #pvalues = np.exp(bentkus_inequality_function(risks,alpha,n))
    
    pvalues = np.minimum(np.exp(hoeffding_inequality_function(risks,alpha,n)),np.exp(bentkus_inequality_function(risks,alpha,n)))
    plot_risks_and_pvalues(risks=risks,pvalues=pvalues,lambdas=lambdas,alpha=alpha,delta=delta,N=N,upper_bound=upper_bound)
    below_delta = (pvalues <= delta/N)
    return risks[below_delta],lambdas[below_delta], pvalues[below_delta]