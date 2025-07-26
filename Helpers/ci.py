from scipy.stats import truncnorm,norm
import pandas as pd
import numpy as np 

def fit_truncated_gaussian(mu, sigma, lower_bound, upper_bound, eps = 1e-10):
    """
    Fits a truncated Gaussian distribution given mean, standard deviation,
    and the lower and upper bounds, and calculates the 95% confidence interval
    and expected value.
    
    Parameters:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        lower_bound (float): The lower boundary of the truncated Gaussian.
        upper_bound (float): The upper boundary of the truncated Gaussian.
        
    Returns:
        dict: A dictionary containing the fitted parameters (mean and std),
              the 95% confidence interval, and the expected value.
    """
    sigma = sigma + eps
    # Scale bounds for truncated normal
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma

    # Fit a truncated Gaussian using scipy's truncnorm
    rv = truncnorm(a, b, loc=mu, scale=sigma)

    # Precompute results
    fitted_mean = rv.mean()
    fitted_std = rv.std()
    confidence_interval = rv.interval(0.95)

    return fitted_mean, fitted_std, confidence_interval

def get_mean_truncated_gaussian(mu, sigma, lower_bound, upper_bound, eps = 1e-10):
    """
    Fits a truncated Gaussian distribution given mean, standard deviation,
    and the lower and upper bounds, and calculates the 95% confidence interval
    and expected value.
    
    Parameters:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        lower_bound (float): The lower boundary of the truncated Gaussian.
        upper_bound (float): The upper boundary of the truncated Gaussian.
        
    Returns:
        dict: A dictionary containing the fitted parameters (mean and std),
              the 95% confidence interval, and the expected value.
    """
    sigma = max(sigma, eps)
    # Scale bounds for truncated normal
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
    return truncnorm.mean(a, b, loc=mu, scale=sigma)



def process_truncated_gaussian_mean(df, scale,lower_bound, upper_bound):
    # Precompute the scaled std values
    scaled_std = df["model_std_values"] * scale
    
    # Apply the fitting function to the DataFrame columns
    results = df.apply(
        lambda row: get_mean_truncated_gaussian(
            np.array(row["model_mean_values"]), np.array(scaled_std[row.name]), lower_bound, upper_bound),
        axis=1
    )

    return results

def get_std_truncated_gaussian(mu, sigma, lower_bound, upper_bound, eps = 1e-10):
    """
    Fits a truncated Gaussian distribution given mean, standard deviation,
    and the lower and upper bounds, and calculates the 95% confidence interval
    and expected value.
    
    Parameters:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        lower_bound (float): The lower boundary of the truncated Gaussian.
        upper_bound (float): The upper boundary of the truncated Gaussian.
        
    Returns:
        dict: A dictionary containing the fitted parameters (mean and std),
              the 95% confidence interval, and the expected value.
    """
    sigma = max(sigma, eps)
    # Scale bounds for truncated normal
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
    return truncnorm.std(a, b, loc=mu, scale=sigma)



def process_truncated_gaussian_std(df, scale,lower_bound, upper_bound):
    # Precompute the scaled std values
    scaled_std = df["model_std_values"] * scale
    
    # Apply the fitting function to the DataFrame columns
    results = df.apply(
        lambda row: get_std_truncated_gaussian(
            np.array(row["model_mean_values"]), np.array(scaled_std[row.name]), lower_bound, upper_bound),
        axis=1
    )

    return results

# Vectorized approach to apply the truncation logic on the DataFrame
def process_truncated_gaussian(df, scale,lower_bound, upper_bound):
    # Precompute the scaled std values
    scaled_std = df["model_std_values"] * scale
    
    # Apply the fitting function to the DataFrame columns
    results = df.apply(
        lambda row: fit_truncated_gaussian(
            np.array(row["model_mean_values"]), scaled_std[row.name], lower_bound, upper_bound),
        axis=1
    )

    # Unpack results into separate lists
    new_mean, new_std, new_ci = zip(*results)

    return pd.DataFrame({
        "mean": new_mean,
        "std": new_std,
        "confidence_interval": new_ci
    })