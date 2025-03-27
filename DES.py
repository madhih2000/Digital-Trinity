import pandas as pd
import streamlit as st
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import re
import time
import fitter
import random
import logging

from scipy.stats import cauchy, chi2, expon, exponpow, gamma, lognorm, norm, powerlaw, rayleigh, uniform
from statsmodels.genmod.families import Poisson, NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from statsmodels.tools import add_constant

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import forecast_models

def load_data(uploaded_file):
    """Loads data from an uploaded Excel file."""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to load and store files in session state
def load_and_store_file(file, key):
    if file is not None and key not in st.session_state:
        st.session_state[key] = load_data(file)


def calculate_safety_stock(dist_name, dist_params, service_level_percentage, std_lead_time):
    """
    Calculates safety stock based on the distribution, its parameters, and the service level.

    Args:
        dist_name (str): The name of the best-fitting distribution.
        dist_params (tuple): The parameters of the distribution.
        service_level_percentage (float): The desired service level percentage.
        std_lead_time (float): The standard deviation of the lead time (used for Normal).

    Returns:
        float: The calculated safety stock.
    """

    z_score = stats.norm.ppf(service_level_percentage / 100)

    if dist_name == "Normal":
        return z_score * std_lead_time
    elif dist_name == "Gamma":
        shape, loc, scale = dist_params
        return stats.gamma.ppf(service_level_percentage / 100, shape, loc=loc, scale=scale) - (shape * scale + loc)
    elif dist_name == "Weibull":
        shape, loc, scale = dist_params
        return stats.weibull_min.ppf(service_level_percentage / 100, shape, loc=loc, scale=scale) - scale * (1 + (np.euler_gamma*shape))
    elif dist_name == "Log-Normal":
        shape, loc, scale = dist_params
        return stats.lognorm.ppf(service_level_percentage / 100, shape, loc=loc, scale=scale) - np.exp(loc + (scale**2)/2)
    elif dist_name == "Exponential":
        loc, scale = dist_params
        return stats.expon.ppf(service_level_percentage / 100, loc=loc, scale=scale) - scale
    elif dist_name == "Beta":
        a, b, loc, scale = dist_params
        return stats.beta.ppf(service_level_percentage / 100, a, b, loc=loc, scale=scale) - (a/(a+b))
    elif dist_name == "Poisson":
        mu = dist_params[0]
        return z_score * np.sqrt(mu)
    elif dist_name == "Negative Binomial":
        mean = dist_params[0]
        var = mean + (dist_params[1] * mean)
        std = np.sqrt(var)
        return z_score * std
    elif dist_name == "Zero-Inflated Poisson":
        mu = dist_params[0][1]
        return z_score * np.sqrt(mu)
    elif dist_name == "Zero-Inflated Negative Binomial":
        mean = dist_params[0][1]
        var = mean + (dist_params[0][0] * mean)
        std = np.sqrt(var)
        return z_score * std
    else:
        return z_score * std_lead_time  # Default to Normal if distribution is unknown


def get_mean_from_distribution(distribution_name, distribution_params):
    """
    Calculate the mean of a distribution based on its parameters.
    """
    if not distribution_params:
        return None  # No valid parameters found

    try:
        if distribution_name == "cauchy":
            loc, _ = distribution_params
            return loc

        elif distribution_name == "chi2":
            df, loc, _ = distribution_params
            return df + loc

        elif distribution_name == "expon":
            loc, scale = distribution_params
            return loc + scale

        elif distribution_name == "exponpow":
            b, loc, scale = distribution_params
            return loc + scale * (1 - np.exp(-1))**(1/b)

        elif distribution_name == "gamma":
            a, loc, scale = distribution_params
            return a * scale + loc

        elif distribution_name == "lognorm":
            s, loc, scale = distribution_params
            return np.exp(np.log(scale) + (s ** 2) / 2) + loc

        elif distribution_name == "norm":
            loc, _ = distribution_params
            return loc

        elif distribution_name == "powerlaw":
            a, loc, scale = distribution_params
            return loc + scale * (a / (a + 1))

        elif distribution_name == "rayleigh":
            loc, scale = distribution_params
            return loc + scale * np.sqrt(np.pi / 2)

        elif distribution_name == "uniform":
            loc, scale = distribution_params
            return loc + scale / 2

        else:
            return None  # Unsupported distribution

    except Exception as e:
        print(f"Error calculating mean: {e}")
        return None


def get_std_from_distribution(distribution_name, distribution_params):
    """
    Calculate the standard deviation of a distribution based on its parameters.
    """
    if not distribution_params:
        return None  # No valid parameters found

    try:
        if distribution_name == "cauchy":
            _, scale = distribution_params
            return scale

        elif distribution_name == "chi2":
            df, _, _ = distribution_params
            return np.sqrt(2 * df)

        elif distribution_name == "expon":
            _, scale = distribution_params
            return scale

        elif distribution_name == "exponpow":
            b, _, scale = distribution_params
            return scale * (1 - np.exp(-1))**(1/b) * np.sqrt(1 - (1 - np.exp(-1))**(2/b))

        elif distribution_name == "gamma":
            a, _, scale = distribution_params
            return np.sqrt(a) * scale

        elif distribution_name == "lognorm":
            s, _, scale = distribution_params
            return np.sqrt((np.exp(s ** 2) - 1) * np.exp(2 * np.log(scale) + s ** 2))

        elif distribution_name == "norm":
            _, sigma = distribution_params
            return sigma

        elif distribution_name == "powerlaw":
            a, _, scale = distribution_params
            return scale * np.sqrt(a / (a + 2)) * np.sqrt(1 - (a / (a + 1))**2)

        elif distribution_name == "rayleigh":
            _, scale = distribution_params
            return scale * np.sqrt((4 - np.pi) / 2)

        elif distribution_name == "uniform":
            _, scale = distribution_params
            return scale / np.sqrt(12)

        else:
            return None  # Unsupported distribution

    except Exception as e:
        print(f"Error calculating standard deviation: {e}")
        return None


def process_lead_time(df):
    #try:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Invalid or empty DataFrame")

    lead_time_values = df.filter(like="Lead Time").iloc[0].dropna().astype(float)

    max_lead_time = lead_time_values.max() if not lead_time_values.empty else np.nan
    std_lead_time = lead_time_values.std(ddof=0) if len(lead_time_values) > 1 else np.nan

    if np.isnan(max_lead_time) or np.isnan(std_lead_time):
        raise ValueError("Computed NaN values")

    best_dist_name, best_dist_params = fit_distribution_lead_time(lead_time_values)

    return round(max_lead_time, 2), round(std_lead_time, 2), best_dist_name, best_dist_params

    # except Exception as e:
    #     print(f"Error: {e}, returning default values.")
    #     return 4, 2, "Normal", (4,2) # Default values and Normal Distribution.


def preprocess_data_consumption(df):
    df.columns = df.columns.str.strip()
    # Step 1: Convert Pstng Date to datetime
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])
    # Drop rows with NaT or NaN in 'Pstng Date' column
    df = df.dropna(subset=['Pstng Date'])
    # Step 2: Extract the week number of the year
    df['Week'] = df['Pstng Date'].dt.isocalendar().week
    # Step 3: Group by Material Number, Plant, Site, and Week, then sum the Quantity
    grouped = df.groupby(['Material Number', 'Plant', 'Site', 'Week'])['Quantity'].sum().reset_index()
    # Step 4: Pivot the data to get quantities per week as columns
    pivot_df = grouped.pivot_table(index=['Material Number', 'Plant', 'Site'], columns='Week', values='Quantity', aggfunc='sum').reset_index()
    # Step 5: Rename the columns to include 'WW' for clarity
    pivot_df.columns = ['Material Number', 'Plant', 'Site'] + [f'WW{int(col)}_Consumption' for col in pivot_df.columns[3:]]
    pivot_df = pivot_df.fillna(0)
    # Apply abs() only to the numeric columns (ignoring non-numeric ones)
    pivot_df.iloc[:, 3:] = pivot_df.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').abs()
    return pivot_df

def preprocess_data_GR(df_GR):
    df_GR.columns = df_GR.columns.str.strip()
    # Step 1: Convert 'Pstng Date' to datetime
    df_GR['Pstng Date'] = pd.to_datetime(df_GR['Pstng Date'], errors='coerce')

    # Extract the week number of the year
    df_GR['Week'] = df_GR['Pstng Date'].dt.isocalendar().week

    # Group by 'Material Number', 'Plant', 'Site', and 'Week', then sum the 'Quantity'
    grouped = df_GR.groupby(['Material Number', 'Plant', 'Site', 'Week'])['Quantity'].sum().reset_index()

    # Pivot the data to get quantities per week as columns
    pivot_df = grouped.pivot_table(index=['Material Number', 'Plant', 'Site'], columns='Week', values='Quantity', aggfunc='sum').reset_index()

    # Fill NaN values with 0 and convert all quantities to positive (absolute value)
    pivot_df = pivot_df.fillna(0)
    pivot_df.iloc[:, 3:] = pivot_df.iloc[:, 3:].abs()  # Assuming columns 3 and onward are the week columns

    # Step 7: Rename the columns to include 'WW' for clarity
    pivot_df.columns = ['Material Number', 'Plant', 'Site'] + [f'WW{int(col)}_GR' for col in pivot_df.columns[3:]]
    return pivot_df

def preprocess_data_OP(df_OR):
    df_OR.columns = df_OR.columns.str.strip()
    # Step 1: Convert 'Pstng Date' to datetime
    df_OR['Document Date'] = pd.to_datetime(df_OR['Document Date'], errors='coerce')

    # Step 2: Count the number of NaN values in 'Pstng Date' (optional)
    nan_count = df_OR['Document Date'].isna().sum()
    print(f"Number of NaN or NaT values in 'Document Date': {nan_count}")

    # Step 3: Extract the week number of the year
    df_OR['Week'] = df_OR['Document Date'].dt.isocalendar().week

    # Step 4: Group by 'Material Number', 'Plant', and 'Week', then sum the 'Order Quantity'
    grouped = df_OR.groupby(['Material Number', 'Plant', 'Week'])['Order Quantity'].sum().reset_index()

    # Step 5: Pivot the data to get quantities per week as columns
    pivot_df = grouped.pivot_table(index=['Material Number', 'Plant'], columns='Week', values='Order Quantity', aggfunc='sum').reset_index()

    # Step 6: Fill NaN values with 0 and convert all quantities to positive (absolute value)
    pivot_df = pivot_df.fillna(0)
    pivot_df.iloc[:, 2:] = pivot_df.iloc[:, 2:].abs()  # Assuming columns 2 and onward are the week columns

    # Step 7: Rename the columns to include 'WW' for clarity
    pivot_df.columns = ['Material Number', 'Plant'] + [f'WW{int(col)}_OP' for col in pivot_df.columns[2:]]
    return pivot_df


def preprocess_data(df, prefix):
    """Preprocesses weekly data columns (WW1, WW2, etc.)."""
    weekly_cols = [col for col in df.columns if col.startswith("WW")]
    if "Site" in df.columns:
        id_vars = ["Material Number", "Plant", "Site"]
        sort_by = ["Material Number", "Plant", "Site", "Week"]
    else:
        id_vars = ["Material Number", "Plant"]
        sort_by = ["Material Number", "Plant", "Week"]

    df_melted = pd.melt(df, id_vars=id_vars, value_vars=weekly_cols, var_name="Week", value_name=prefix)
    df_melted["Week"] = df_melted["Week"].apply(lambda x: abs(int(re.findall(r'\d+', x)[0])))
    df_melted = df_melted.sort_values(by=sort_by)
    return df_melted

def find_best_distribution(data, include_zero_inflated=False, include_hurdle=False):
    # Ensure data is a numpy array
    data = np.array(data)
    
    # Remove non-positive values for continuous distributions
    data_positive = data[data > 0]
    
    # Check for zero-inflation
    zero_fraction = np.sum(data == 0) / len(data)
    
    # Fit distributions using fitter
    f = fitter.Fitter(data_positive, distributions=fitter.get_common_distributions())
    f.fit()
    
    # Get the summary table
    summary = f.summary()
    summary.index.names = ['name']
    summary = summary.reset_index()
    
    # Extract AIC and BIC values from the summary table
    aic_values = summary['aic']
    bic_values = summary['bic']
    
    # Find the distribution with the lowest AIC and BIC
    best_distribution_name_aic = summary['name'][aic_values.idxmin()]
    best_distribution_name_bic = summary['name'][bic_values.idxmin()]
    
    # Get the parameters for the best distributions
    best_distribution_params_aic = f.fitted_param[best_distribution_name_aic]
    best_distribution_params_bic = f.fitted_param[best_distribution_name_bic]
    
    # Compare AIC and BIC values to choose the best distribution
    if aic_values.min() < bic_values.min():
        best_distribution_name = best_distribution_name_aic
        best_distribution_params = best_distribution_params_aic
    else:
        best_distribution_name = best_distribution_name_bic
        best_distribution_params = best_distribution_params_bic
    
    return best_distribution_name, best_distribution_params
    

# def find_best_distribution(data, include_zero_inflated=False, include_hurdle=False):
#     # Ensure data is a numpy array
#     data = np.array(data)
    
#     # Remove non-positive values for continuous distributions
#     data_positive = data[data > 0]
    
#     # Check for zero-inflation
#     zero_fraction = np.sum(data == 0) / len(data)
    
#     # Define the distributions to be tested
#     distributions = {
#         'Normal': stats.norm,
#         'Gamma': stats.gamma,
#         'Weibull': stats.weibull_min,
#         'Log-Normal': stats.lognorm,
#         'Exponential': stats.expon,
#         'Beta': stats.beta
#     }
    
#     best_distribution = None
#     best_aic = float('inf')
    
#     ### **1. Fit Discrete Distributions First**
#     # Poisson Distribution
#     try:
#         mu = np.mean(data)  # Poisson parameter (mean)
#         log_likelihood = np.sum(stats.poisson.logpmf(data, mu))
#         poisson_aic = -2 * log_likelihood + 2  # 1 parameter (mu)
        
#         if poisson_aic < best_aic:
#             best_aic = poisson_aic
#             best_distribution = 'Poisson'
#     except Exception as e:
#         st.error(f"Error fitting Poisson: {e}")
    
#     # Negative Binomial Distribution
#     try:
#         nb_model = sm.GLM(data, np.ones(len(data)), family=sm.families.NegativeBinomial()).fit()
#         nb_aic = nb_model.aic
        
#         if nb_aic < best_aic:
#             best_aic = nb_aic
#             best_distribution = 'Negative Binomial'
#     except Exception as e:
#         st.error(f"Error fitting Negative Binomial: {e}")
    
#     ### **2. Fit Continuous Distributions (if necessary)**
#     for name, distribution in distributions.items():
#         try:
#             if name == 'Log-Normal':
#                 # For Log-Normal, fit to the log of the data
#                 params = distribution.fit(np.log(data_positive))
#             else:
#                 params = distribution.fit(data_positive)
            
#             if name == 'Log-Normal':
#                 # For Log-Normal, calculate log-likelihood on the log-transformed data
#                 log_likelihood = np.sum(distribution.logpdf(np.log(data_positive), *params))
#             else:
#                 log_likelihood = np.sum(distribution.logpdf(data_positive, *params))
            
#             aic = -2 * log_likelihood + 2 * len(params)
            
#             if aic < best_aic:
#                 best_aic = aic
#                 best_distribution = name
#         except Exception as e:
#             st.error(f"Error fitting {name}: {e}")
    
#     ### **3. Fit Zero-Inflated Models (if enabled)**
#     if include_zero_inflated and zero_fraction > 0.2:  # Only consider if >20% zeros
#         try:
#             exog = add_constant(np.ones(len(data)))  # Exogenous variable
            
#             # Zero-Inflated Poisson
#             zip_model = ZeroInflatedPoisson(data, exog).fit(disp=0)
#             zip_aic = zip_model.aic
            
#             if zip_aic < best_aic:
#                 best_aic = zip_aic
#                 best_distribution = "Zero-Inflated Poisson"
#         except Exception as e:
#             st.error(f"Error fitting Zero-Inflated Poisson: {e}")
        
#         try:
#             # Zero-Inflated Negative Binomial
#             zinb_model = ZeroInflatedNegativeBinomialP(data, exog).fit(disp=0)
#             zinb_aic = zinb_model.aic
            
#             if zinb_aic < best_aic:
#                 best_aic = zinb_aic
#                 best_distribution = "Zero-Inflated Negative Binomial"
#         except Exception as e:
#             st.error(f"Error fitting Zero-Inflated Negative Binomial: {e}")
    
#     return best_distribution

# def simulate_demand(fitted_distribution_params, num_simulations=10000):
#     """
#     Simulate demand based on the fitted distribution parameters.
#     """
#     distribution_type = fitted_distribution_params['distribution']

#     if distribution_type == "Normal":
#         mu, std = fitted_distribution_params['mu'], fitted_distribution_params['std']
#         simulated_demand = np.random.normal(mu, std, num_simulations)

#     elif distribution_type == "Poisson":
#         mu = fitted_distribution_params['mu']
#         simulated_demand = np.random.poisson(mu, num_simulations)

#     elif distribution_type == "Negative Binomial":
#         n, p = fitted_distribution_params['n'], fitted_distribution_params['p']
#         simulated_demand = np.random.negative_binomial(n, p, num_simulations)

#     elif distribution_type == "Gamma":
#         a, loc, scale = fitted_distribution_params['a'], fitted_distribution_params['loc'], fitted_distribution_params['scale']
#         simulated_demand = np.random.gamma(a, scale, num_simulations)

#     elif distribution_type == "Weibull":
#         c, loc, scale = fitted_distribution_params['c'], fitted_distribution_params['loc'], fitted_distribution_params['scale']
#         simulated_demand = np.random.weibull(c, num_simulations) * scale + loc

#     elif distribution_type == "Log-Normal":
#         s, loc, scale = fitted_distribution_params['s'], fitted_distribution_params['loc'], fitted_distribution_params['scale']
#         simulated_demand = np.random.lognormal(np.log(scale), s, num_simulations)

#     elif distribution_type == "Exponential":
#         loc, scale = fitted_distribution_params['loc'], fitted_distribution_params['scale']
#         simulated_demand = np.random.exponential(scale, num_simulations)

#     elif distribution_type == "Beta":
#         a, b, loc, scale = fitted_distribution_params['a'], fitted_distribution_params['b'], fitted_distribution_params['loc'], fitted_distribution_params['scale']
#         simulated_demand = np.random.beta(a, b, num_simulations) * (scale - loc) + loc

#     else:
#         st.warning("Unsupported distribution for simulation.")
#         return None

#     return simulated_demand

def simulate_demand(fitted_distribution_name, fitted_distribution_params, num_simulations=10000):
    """
    Simulate demand based on the fitted distribution parameters.
    """
    try:
        # Generate random samples from the specified distribution
        if fitted_distribution_name == "cauchy":
            loc, scale = fitted_distribution_params
            simulated_demand = stats.cauchy.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "chi2":
            df, loc, scale = fitted_distribution_params
            simulated_demand = stats.chi2.rvs(df=df, loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "expon":
            loc, scale = fitted_distribution_params
            simulated_demand = stats.expon.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "exponpow":
            b, loc, scale = fitted_distribution_params
            simulated_demand = stats.exponpow.rvs(b=b, loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "gamma":
            a, loc, scale = fitted_distribution_params
            simulated_demand = stats.gamma.rvs(a=a, loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "lognorm":
            s, loc, scale = fitted_distribution_params
            simulated_demand = stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "norm":
            loc, scale = fitted_distribution_params
            simulated_demand = stats.norm.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "powerlaw":
            a, loc, scale = fitted_distribution_params
            simulated_demand = stats.powerlaw.rvs(a=a, loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "rayleigh":
            loc, scale = fitted_distribution_params
            simulated_demand = stats.rayleigh.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif fitted_distribution_name == "uniform":
            loc, scale = fitted_distribution_params
            simulated_demand = stats.uniform.rvs(loc=loc, scale=scale, size=num_simulations)
        
        else:
            st.warning(f"Unsupported distribution for simulation: {fitted_distribution_name}")
            return None

        return simulated_demand

    except Exception as e:
        st.error(f"Error simulating demand: {e}")
        return None

def fit_distribution(data_values, data_type="Consumption"):
    """
    Finds the best fitting distribution for the given data values and returns the parameters,
    ensuring non-negative values where appropriate.
    """
    try:
        # Ensure data is a numpy array
        data = np.array(data_values)
        
        # Remove non-positive values for continuous distributions
        data_positive = data[data > 0]
        
        # Fit distributions using fitter
        f = fitter.Fitter(data_positive, distributions=fitter.get_common_distributions())
        f.fit()
        
        # Get the summary table
        summary = f.summary()
        summary.index.names = ['name']
        summary = summary.reset_index()
        
        # Extract AIC and BIC values from the summary table
        aic_values = summary['aic']
        bic_values = summary['bic']
        
        # Find the distribution with the lowest AIC and BIC
        best_distribution_name_aic = summary['name'][aic_values.idxmin()]
        best_distribution_name_bic = summary['name'][bic_values.idxmin()]
        
        # Get the parameters for the best distributions
        best_distribution_params_aic = f.fitted_param[best_distribution_name_aic]
        best_distribution_params_bic = f.fitted_param[best_distribution_name_bic]
        
        # Compare AIC and BIC values to choose the best distribution
        if aic_values.min() < bic_values.min():
            best_distribution_name = best_distribution_name_aic
            best_distribution_params = best_distribution_params_aic
        else:
            best_distribution_name = best_distribution_name_bic
            best_distribution_params = best_distribution_params_bic
        
        best_distribution_params = tuple(map(float, best_distribution_params))
        # Display the best distribution and parameters
        st.success(f"Best {data_type} Distribution: {best_distribution_name} with parameters: {best_distribution_params}")
        
        return best_distribution_params, best_distribution_name

    except Exception as e:
        st.error(f"Error finding best distribution for {data_type}: {e}")
        return None, None

# def fit_distribution(data_values, data_type="Consumption"):
#     """
#     Finds the best fitting distribution for the given data values and returns the parameters,
#     ensuring non-negative values where appropriate.
#     """
#     best_distribution = find_best_distribution(data_values, include_zero_inflated=True, include_hurdle=True)

#     if not best_distribution:
#         st.warning(f"Could not find a suitable distribution for {data_type}.")
#         return None

#     try:
#         distribution_params = {'distribution': best_distribution}

#         if best_distribution == "Normal":
#             mu, std = stats.norm.fit(data_values)
#             std = max(std, 0)  # Ensure std is non-negative
#             distribution_params.update({'mu': mu, 'std': std})
#             st.success(f"Best {data_type} Distribution: Normal (Mean = {mu:.2f}, Std Dev = {std:.2f})")

#         elif best_distribution == "Poisson":
#             mu = max(stats.poisson.fit(data_values)[0], 0)  # Ensure non-negative mean
#             distribution_params.update({'mu': mu})
#             st.success(f"Best {data_type} Distribution: Poisson (Mean = {mu:.2f})")

#         elif best_distribution == "Negative Binomial":
#             n, p, loc = stats.nbinom.fit(data_values)
#             n, p = max(n, 0), max(p, 0)  # Ensure non-negative parameters
#             distribution_params.update({'n': n, 'p': p})
#             st.success(f"Best {data_type} Distribution: Negative Binomial (n = {n:.2f}, p = {p:.2f})")

#         elif best_distribution == "Gamma":
#             a, loc, scale = stats.gamma.fit(data_values)
#             a = max(a, 1)
#             a, loc, scale = max(a, 0), max(loc, 0), max(scale, 0)
#             distribution_params.update({'a': a, 'loc': loc, 'scale': scale})
#             st.success(f"Best {data_type} Distribution: Gamma (a = {a:.2f}, loc = {loc:.2f}, scale = {scale:.2f})")

#         elif best_distribution == "Weibull":
#             c, loc, scale = stats.weibull_min.fit(data_values)
#             c, loc, scale = max(c, 0), max(loc, 0), max(scale, 0)
#             distribution_params.update({'c': c, 'loc': loc, 'scale': scale})
#             st.success(f"Best {data_type} Distribution: Weibull (c = {c:.2f}, loc = {loc:.2f}, scale = {scale:.2f})")

#         elif best_distribution == "Log-Normal":
#             s, loc, scale = stats.lognorm.fit(data_values)
#             distribution_params.update({'s': s, 'loc': loc, 'scale': scale})
#             st.success(f"Best {data_type} Distribution: Log-Normal (s = {s:.2f}, loc = {loc:.2f}, scale = {scale:.2f})")

#         elif best_distribution == "Exponential":
#             loc, scale = stats.expon.fit(data_values)
#             loc, scale = max(loc, 0), max(scale, 0)
#             distribution_params.update({'loc': loc, 'scale': scale})
#             st.success(f"Best {data_type} Distribution: Exponential (loc = {loc:.2f}, scale = {scale:.2f})")

#         elif best_distribution == "Beta":
#             a, b, loc, scale = stats.beta.fit(data_values)
#             a, b, loc, scale = max(a, 0), max(b, 0), max(loc, 0), max(scale, 0)
#             distribution_params.update({'a': a, 'b': b, 'loc': loc, 'scale': scale})
#             st.success(f"Best {data_type} Distribution: Beta (a = {a:.2f}, b = {b:.2f}, loc = {loc:.2f}, scale = {scale:.2f})")

#         elif best_distribution in ["Zero-Inflated Poisson", "Zero-Inflated Negative Binomial", "Hurdle Poisson"]:
#             st.success(f"Best {data_type} Distribution: {best_distribution}")

#         else:
#             st.warning(f"Could not find a suitable distribution for {data_type}.")
#             return None

#         return distribution_params, best_distribution

#     except Exception as e:
#         st.error(f"Error fitting distributions for {data_type}: {e}")
#         return None, None


def fit_distribution_lead_time(data_values, data_type="Lead Time"):
    """
    Finds the best fitting distribution for the given data values and returns the parameters,
    ensuring non-negative values where appropriate.
    """
    try:
        best_distribution_name, best_distribution_params = find_best_distribution(data_values, include_zero_inflated=True, include_hurdle=True)
    except Exception as e:
        st.error(f"Error finding best distribution for {data_type}: {e}")
        return None, None

    if not best_distribution_name:
        st.warning(f"Could not find a suitable distribution for {data_type}.")
        return None, None
    return best_distribution_params, best_distribution_name


def simulate_consumption(consumption_distribution_name, consumption_distribution_params, num_simulations=1):
    """
    Simulate consumption based on the fitted distribution parameters.
    """
    try:
        # Generate random samples from the specified distribution
        if consumption_distribution_name == "cauchy":
            loc, scale = consumption_distribution_params
            simulated_consumption = cauchy.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "chi2":
            df, loc, scale = consumption_distribution_params
            simulated_consumption = chi2.rvs(df=df, loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "expon":
            loc, scale = consumption_distribution_params
            simulated_consumption = expon.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "exponpow":
            b, loc, scale = consumption_distribution_params
            simulated_consumption = exponpow.rvs(b=b, loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "gamma":
            a, loc, scale = consumption_distribution_params
            simulated_consumption = gamma.rvs(a=a, loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "lognorm":
            s, loc, scale = consumption_distribution_params
            simulated_consumption = lognorm.rvs(s=s, loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "norm":
            loc, scale = consumption_distribution_params
            simulated_consumption = norm.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "powerlaw":
            a, loc, scale = consumption_distribution_params
            simulated_consumption = powerlaw.rvs(a=a, loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "rayleigh":
            loc, scale = consumption_distribution_params
            simulated_consumption = rayleigh.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif consumption_distribution_name == "uniform":
            loc, scale = consumption_distribution_params
            simulated_consumption = uniform.rvs(loc=loc, scale=scale, size=num_simulations)
        else:
            print(f"Unsupported distribution for simulation: {consumption_distribution_name}")
            return None

        return simulated_consumption

    except Exception as e:
        print(f"Error simulating consumption: {e}")
        return None

def simulate_ordering(order_distribution_name, order_distribution_params, num_simulations=1):
    """
    Simulate ordering based on the fitted distribution parameters.
    """
    try:
        # Generate random samples from the specified distribution
        if order_distribution_name == "cauchy":
            loc, scale = order_distribution_params
            simulated_ordering = cauchy.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "chi2":
            df, loc, scale = order_distribution_params
            simulated_ordering = chi2.rvs(df=df, loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "expon":
            loc, scale = order_distribution_params
            simulated_ordering = expon.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "exponpow":
            b, loc, scale = order_distribution_params
            simulated_ordering = exponpow.rvs(b=b, loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "gamma":
            a, loc, scale = order_distribution_params
            simulated_ordering = gamma.rvs(a=a, loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "lognorm":
            s, loc, scale = order_distribution_params
            simulated_ordering = lognorm.rvs(s=s, loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "norm":
            loc, scale = order_distribution_params
            simulated_ordering = norm.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "powerlaw":
            a, loc, scale = order_distribution_params
            simulated_ordering = powerlaw.rvs(a=a, loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "rayleigh":
            loc, scale = order_distribution_params
            simulated_ordering = rayleigh.rvs(loc=loc, scale=scale, size=num_simulations)
        
        elif order_distribution_name == "uniform":
            loc, scale = order_distribution_params
            simulated_ordering = uniform.rvs(loc=loc, scale=scale, size=num_simulations)
        
        else:
            print(f"Unsupported distribution for simulation: {order_distribution_name}")
            return None

        return simulated_ordering

    except Exception as e:
        print(f"Error simulating ordering: {e}")
        return None

# Inventory Simulation
def simulate_inventory(filtered_consumption, filtered_orders, filtered_receipts, initial_inventory, reorder_point, order_quantity, lead_time, lead_time_std_dev, demand_surge_weeks, demand_surge_factor, consumption_distribution_params, consumption_type, consumption_best_distribution, consumption_values, num_weeks, order_distribution_params,order_distribution_best, order_quantity_type):

    inventory = initial_inventory
    orders_pending = {}
    inventory_history = []
    stockout_weeks = []
    wos_history = []

    proactive_inventory = initial_inventory
    proactive_orders_pending = {}
    proactive_inventory_history = []
    proactive_stockout_weeks = []
    proactive_wos_history = []

    consumption_history = []
    weeks = list(range(1, num_weeks + 1))
    weekly_events = []
    logging.warning("Simulation started.")

    for i, week in enumerate(weeks):
        logging.warning(f"Week {week} - Starting Inventory (Reactive): {inventory}, (Proactive): {proactive_inventory}")
        event_description = f"**Week {week}**\n"
        event_description += f"Starting Inventory (Reactive): {inventory}\n"
        event_description += f"Starting Inventory (Proactive): {proactive_inventory}\n"

        # Add receipts
        if week in orders_pending:
            inventory += orders_pending[week]
            logging.warning(f"Reactive Order of {orders_pending[week]} arrived.")
            event_description += f"Reactive Order of {orders_pending[week]} arrived.\n"
            del orders_pending[week]

        if week in proactive_orders_pending:
            proactive_inventory += proactive_orders_pending[week]
            logging.warning(f"Proactive Order of {proactive_orders_pending[week]} arrived.")
            event_description += f"Proactive Order of {proactive_orders_pending[week]} arrived.\n"
            del proactive_orders_pending[week]


        # Consumption
        consumption_source = "Fixed" if consumption_type == "Fixed" else f"Distribution ({consumption_best_distribution} with parameters {consumption_distribution_params})" if consumption_distribution_params else "Unknown Distribution"
        if consumption_type == "Fixed":
            consumption_this_week = consumption_values[i] if i < len(consumption_values) else 0 # Handle if user provides less consumption values than weeks.
        elif consumption_type == "Distribution" and consumption_distribution_params:
            consumption_this_week = simulate_consumption(consumption_best_distribution, consumption_distribution_params)
            if consumption_this_week is None:
                consumption_this_week = 0
        else:
            consumption_this_week = 0

        consumption_this_week = int(consumption_this_week)
        # Apply demand surge (override distribution)
        if f"WW{i + 1}" in demand_surge_weeks: 
            consumption_this_week = consumption_this_week * demand_surge_factor
            logging.warning(f"Demand surge applied: Consumption increased to {consumption_this_week}.")
            event_description += f"Demand surge applied. Consumption increased by {demand_surge_factor}x.\n"


        # Deduct consumption
        inventory -= consumption_this_week
        if inventory < 0:
            stockout_weeks.append(week)
            inventory = 0
            logging.warning(f"Stockout occurred in week {week}.")
            event_description += "Stockout occurred.\n"

        # Proactive inventory deduction
        proactive_inventory -= consumption_this_week
        if proactive_inventory < 0:
            proactive_stockout_weeks.append(week)
            proactive_inventory = 0
            logging.warning(f"Proactive stockout occurred in week {week}.")
            event_description += "Proactive stockout occurred.\n"

        
        consumption_history.append(consumption_this_week)
        logging.warning(f"Consumption this week: {consumption_this_week}")
        event_description += f"Consumption this week: {consumption_this_week} (Source: {consumption_source})\n"
        consumption_df_for_forecasting = pd.DataFrame({
            'Year': [2025] * len(consumption_history),
            'Week': [i + 1 for i in range(len(consumption_history))],
            'Consumption': consumption_history
        })

        forecast_results_df = forecast_models.forecast_weekly_consumption_xgboost_v3(filtered_consumption, consumption_df_for_forecasting, int(lead_time * 1.5))
        forecasted_values = forecast_results_df.predicted_consumption.values
        forecasted_values = forecasted_values[:-1]
        sum_of_forecasted_values = int(forecasted_values.sum())
        logging.warning(f"Forecasted consumption for next {lead_time} weeks: {sum_of_forecasted_values}")
        event_description += f"Forecasted consumption for next {lead_time} weeks is {sum_of_forecasted_values}.\n"

        proactive_forecast = False
        variation = round(random.gauss(0, lead_time_std_dev))
        # Check for reorder
        if proactive_inventory  <= sum_of_forecasted_values:
            order_quantity_to_use = sum_of_forecasted_values - proactive_inventory
            order_quantity_to_use = max(1, int(order_quantity_to_use))  # Ensure minimum order of 1

            order_arrival = int(i + lead_time + variation)
            if order_arrival < num_weeks:
                proactive_orders_pending[weeks[order_arrival]] = order_quantity_to_use
                logging.warning(f"Proactive Order of {order_quantity_to_use} placed for week {weeks[order_arrival]}.")
                event_description += f"Proactive Order of {order_quantity_to_use} placed due to forecasted consumption. Arrival in week {weeks[order_arrival]}.\n"
                proactive_forecast = True
        else:
            event_description += "No Proactive Order placed due to forecasting this week.\n"

        # if proactive_inventory <= reorder_point and not proactive_forecast:
        #     order_quantity_to_use = order_quantity
        #     order_values = filtered_orders.iloc[:, 3:].values.flatten()
        #     if order_quantity_type == "Distribution" and order_distribution_params:
        #         order_quantity_to_use = simulate_ordering(order_distribution_best, order_distribution_params)
        #         if order_quantity_to_use is None:
        #             order_quantity_to_use = 0
            
        #     average_consumption = np.max(order_values)
        #     order_quantity_to_use = min(average_consumption, int(order_quantity_to_use))
        #     order_arrival = int(i + lead_time + round(random.gauss(0, lead_time_std_dev)))

        #     if order_arrival < num_weeks:
        #         proactive_orders_pending[weeks[order_arrival]] = order_quantity_to_use
        #         event_description += f" Proactive Order of {order_quantity_to_use} placed due to reorder point. Arrival in week {weeks[order_arrival]}.\n"
        # else:
        #     event_description += "No proactive order placed this week.\n"

        # Check for reorder
        if (inventory <= reorder_point) or (proactive_inventory <= reorder_point and not proactive_forecast):
            order_quantity_to_use = order_quantity
            order_values = filtered_orders.iloc[:, 3:].values.flatten()
            if order_quantity_type == "Distribution" and order_distribution_params:
                order_quantity_to_use = simulate_ordering(order_distribution_best, order_distribution_params)
                if order_quantity_to_use is None:
                    order_quantity_to_use = 0
            average_consumption = np.max(order_values)
            order_quantity_to_use = min(average_consumption, int(order_quantity_to_use))
            order_arrival = int(i + lead_time + variation)

            if order_arrival < num_weeks:
                orders_pending[weeks[order_arrival]] = order_quantity_to_use
                
                # Check if the proactive condition is met before adding to proactive_orders_pending
                if proactive_inventory <= reorder_point and not proactive_forecast:
                    proactive_orders_pending[weeks[order_arrival]] = order_quantity_to_use
                    logging.warning(f"Proactive Order of {order_quantity_to_use} placed for week {weeks[order_arrival]} due to reorder point.")
                    event_description += f"Proactive Order of {order_quantity_to_use} placed due to reorder point. Arrival in week {weeks[order_arrival]}.\n"
                
                event_description += f"Reactive Order of {order_quantity_to_use} placed due to reorder point. Arrival in week {weeks[order_arrival]}.\n"
        else:
            event_description += "No Reactive Order placed this week.\n"

        # Calculate WoS
        average_consumption = sum(consumption_history[:i + 1]) / (i + 1) if i >= 0 else 0
        wos = inventory / average_consumption if average_consumption > 0 else 0
        wos_history.append(wos)

        proactive_average_consumption = average_consumption
        proactive_wos = proactive_inventory / proactive_average_consumption if proactive_average_consumption > 0 else 0
        proactive_wos_history.append(proactive_wos)

        inventory_history.append(inventory)
        proactive_inventory_history.append(proactive_inventory)
        event_description += f"Reactive Ending Inventory: {inventory}\n"
        event_description += f"Proactive Ending Inventory: {proactive_inventory}\n"
        logging.warning(f"Week {week} - Ending Inventory (Reactive): {inventory}, (Proactive): {proactive_inventory}")
        event_description += "---\n"
        weekly_events.append(event_description)
    
    logging.warning("Simulation completed.")
    return inventory_history, proactive_inventory_history, stockout_weeks, proactive_stockout_weeks, wos_history, proactive_wos_history, consumption_history, weekly_events

def run_monte_carlo_simulation(N, *args):
    all_inventory_histories = []
    all_proactive_inventory_histories = []
    all_stockout_weeks = []
    all_proactive_stockout_weeks = []
    all_wos_histories = []
    all_proactive_wos_histories = []
    all_consumption_histories = []
    all_weekly_events = []

    # Create a progress bar
    progress_text = "Processing simulation: 1 out of {N}"
    my_bar = st.progress(0, text=progress_text.format(N=N))

    for i in range(N):
        time.sleep(0.01)
        inventory_history, proactive_inventory_history, stockout_weeks, proactive_stockout_weeks, wos_history, proactive_wos_history, consumption_history, weekly_events = simulate_inventory(*args)

        all_inventory_histories.append(inventory_history)
        all_proactive_inventory_histories.append(proactive_inventory_history)
        all_stockout_weeks.append(stockout_weeks)
        all_proactive_stockout_weeks.append(proactive_stockout_weeks)
        all_wos_histories.append(wos_history)
        all_proactive_wos_histories.append(proactive_wos_history)
        all_consumption_histories.append(consumption_history)
        all_weekly_events.append(weekly_events)

        # Update progress bar
        my_bar.progress((i + 1) / N, text=f"Processing simulation: {i + 2} out of {N}")
        
    time.sleep(1)
    # Remove progress bar when done
    my_bar.empty()

    return (
        all_inventory_histories,
        all_proactive_inventory_histories,
        all_stockout_weeks,
        all_proactive_stockout_weeks,
        all_wos_histories,
        all_proactive_wos_histories,
        all_consumption_histories,
        all_weekly_events
    )

# def compute_averages(all_inventory_histories, all_stockout_weeks, all_wos_histories, all_consumption_histories):
#     avg_inventory = np.mean(all_inventory_histories, axis=0)
#     avg_wos = np.mean(all_wos_histories, axis=0)
#     avg_consumption = np.mean(all_consumption_histories, axis=0)

#     # Stockout frequency: Percentage of runs where a stockout occurred in each week
#     stockout_frequency = np.mean([len(stockout_weeks) > 0 for stockout_weeks in all_stockout_weeks])

#     return avg_inventory, avg_wos, avg_consumption, stockout_frequency

def compute_averages(all_inventory_histories, all_proactive_inventory_histories, all_stockout_weeks, all_proactive_stockout_weeks, all_wos_histories, all_proactive_wos_histories, all_consumption_histories):
    avg_inventory = np.mean(all_inventory_histories, axis=0)
    avg_proactive_inventory = np.mean(all_proactive_inventory_histories, axis=0)
    avg_wos = np.mean(all_wos_histories, axis=0)
    avg_proactive_wos = np.mean(all_proactive_wos_histories, axis=0)
    avg_consumption = np.mean(all_consumption_histories, axis=0)

    # Stockout frequency: Percentage of runs where a stockout occurred in each week
    stockout_frequency = np.mean([len(stockout_weeks) > 0 for stockout_weeks in all_stockout_weeks])
    stockout_frequency_proactive = np.mean([len(stockout_weeks) > 0 for stockout_weeks in all_proactive_stockout_weeks])
    return avg_inventory, avg_wos, avg_consumption, stockout_frequency, avg_proactive_inventory,avg_proactive_wos, stockout_frequency_proactive

def find_representative_run(all_inventory_histories, avg_inventory):
    distances = []
    for inventory_history in all_inventory_histories:
        distance = np.linalg.norm(np.array(inventory_history) - np.array(avg_inventory))
        distances.append(distance)

    # Find the run with the smallest distance
    representative_index = np.argmin(distances)
    return representative_index


def get_representative_run_details(representative_index, all_inventory_histories, all_proactive_inventory_histories, all_stockout_weeks, all_proactive_stockout_weeks, all_wos_histories, all_proactive_wos_histories, all_consumption_histories, all_weekly_events):
    return (
        all_inventory_histories[representative_index],
        all_proactive_inventory_histories[representative_index],
        all_stockout_weeks[representative_index],
        all_proactive_stockout_weeks[representative_index],
        all_wos_histories[representative_index],
        all_proactive_wos_histories[representative_index],
        all_consumption_histories[representative_index],
        all_weekly_events[representative_index]

        )

def extract_weekly_table(weekly_events):
    data = []

    for i, event in enumerate(weekly_events, start=1):
        lines = event.split("\n")
        week_data = {
            'Week Number': i,
            'Initial Inventory (Reactive)': 0,
            'Initial Inventory (Proactive)': 0,
            'Predicted Consumption for Next Few Weeks (Sum)': 0,
            'Consumption': 0,
            'End of Week Inventory (Proactive)': 0,
            'End of Week Inventory (Reactive)': 0,
            'Proactive Order Placed': 0,
            'Reactive Order Placed': 0,
            'Stockout': False  
        }

        for line in lines:
            if "Starting Inventory (Reactive)" in line:
                try:
                    week_data['Initial Inventory (Reactive)'] = int(float(re.search(r": (\d+\.?\d*)", line).group(1)))
                except (ValueError, AttributeError):
                    print(f"Error extracting 'Starting Inventory (Reactive)' from line: {line}")
            elif "Starting Inventory (Proactive)" in line:
                try:
                    week_data['Initial Inventory (Proactive)'] = int(float(re.search(r": (\d+\.?\d*)", line).group(1)))
                except (ValueError, AttributeError):
                    print(f"Error extracting 'Starting Inventory (Proactive)' from line: {line}")
            elif "Forecasted consumption for next" in line:
                try:
                    forecasted_value = re.search(r"is (\d+\.?\d*)", line).group(1)
                    week_data['Predicted Consumption for Next Few Weeks (Sum)'] = int(float(forecasted_value))
                except (ValueError, AttributeError):
                    print(f"Error extracting 'Forecasted consumption for next' from line: {line}")
            elif "Consumption this week" in line:
                try:
                    week_data['Consumption'] = int(float(re.search(r": (\d+\.?\d*)", line).group(1)))
                except (ValueError, AttributeError):
                    print(f"Error extracting 'Consumption this week' from line: {line}")
            elif "Proactive Ending Inventory" in line:
                try:
                    week_data['End of Week Inventory (Proactive)'] = int(float(re.search(r": (\d+\.?\d*)", line).group(1)))
                except (ValueError, AttributeError):
                    print(f"Error extracting 'Proactive Ending Inventory' from line: {line}")
            elif "Reactive Ending Inventory" in line:
                try:
                    week_data['End of Week Inventory (Reactive)'] = int(float(re.search(r": (\d+\.?\d*)", line).group(1)))
                except (ValueError, AttributeError):
                    print(f"Error extracting 'Reactive Ending Inventory' from line: {line}")
            elif "Proactive Order of" in line and "placed" in line:
                try:
                    order_value = re.search(r"Proactive Order of (\d+\.?\d*)", line).group(1)
                    week_data['Proactive Order Placed'] = int(float(order_value))
                except (ValueError, AttributeError):
                    print(f"Error extracting 'Proactive Order of' from line: {line}")
            elif "Reactive Order of" in line and "placed" in line:
                try:
                    order_value = re.search(r"Reactive Order of (\d+\.?\d*)", line).group(1)
                    week_data['Reactive Order Placed'] = int(float(order_value))
                except (ValueError, AttributeError):
                    print(f"Error extracting 'Reactive Order of' from line: {line}")
            elif "Stockout occurred" in line:
                week_data['Stockout'] = True

        data.append(week_data)

    df = pd.DataFrame(data)
    df.set_index('Week Number', inplace=True) 
    return df