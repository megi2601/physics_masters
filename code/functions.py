import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import norm

def load_data_pol(dir):
    dataframes = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        name = filename.replace('.prn', '')
        if 'WIG' in name:
            continue
        df = pd.read_csv(path, delimiter=',', header=0, parse_dates=['Date'], usecols=['Date', 'Close'])
        df.set_index('Date', inplace=True)
        df.rename(columns={"Close": name}, inplace=True)

        dataframes.append(df)
    return dataframes

def load_data_foreign(index, data_parent_dir):
    data_subdir = os.path.join(data_parent_dir, index)
    dataframes = []
    for filename in os.listdir(data_subdir):
        path = os.path.join(data_subdir, filename)
        name = filename.replace('.csv', '')

        df = pd.read_csv(path, delimiter=',', skiprows=3, header=0, names = ['Date', 'Close','High','Low','Open','Volume'], parse_dates=['Date'], usecols=['Date', 'Close'])
        df.set_index('Date', inplace=True)
        df.rename(columns={"Close": name}, inplace=True)

        dataframes.append(df)
    return dataframes

# def concat_dfs_by_date(dataframes, len_quantile = 0.85, statr_year = 2010, outliers_quantiles = (0.995, 0.005)):
#     df_concat = pd.concat(dataframes, axis=1)
#     df_concat.index = df_concat.index.astype(dataframes[0].index.dtype)
    
#     if statr_year:
#         # only stocks withing the desired time period
#         df_concat = df_concat[statr_year:] #from start_year
#         df_concat = df_concat.loc[:, (df_concat.iloc[-1].notna() | df_concat.iloc[-2].notna())] #to present

#     if len_quantile:
#         # only stocks with many time points
#         q = df_concat.count().quantile(len_quantile).astype(int)
#         df_concat = df_concat.loc[:, df_concat.count() >= q]

#     # calculate daily returns
#     df_concat = df_concat.pct_change(fill_method=None) # confirmed that this should not be diff
    
#     if outliers_quantiles:
#         # remove outliers
#         qhigh = df_concat.quantile(outliers_quantiles[0], axis=0)
#         qlow = df_concat.quantile(outliers_quantiles[1], axis=0)
        
#         for col in df_concat.columns:
#             df_concat.loc[df_concat[col] > qhigh[col], col] = np.nan
#             df_concat.loc[df_concat[col] < qlow[col], col] = np.nan

#     # fill any missing values with last known value
#     df_concat.ffill(inplace=True)
    
#     df_concat = (df_concat - df_concat.mean()) /df_concat.std() # normalization
#     return df_concat

def concat_dfs_by_date(dataframes, min_non_na_fraction_col = 0.85, start_year = '2010', extreme_return_abs = 5, last_traded = True, min_non_na_fraction_row = 0.6, smooth_local_volatility = True):
    df_concat = pd.concat(dataframes, axis=1)
    df_concat.index = df_concat.index.astype(dataframes[0].index.dtype)
    df_concat.sort_index(inplace=True)

    # YEAR RANGE
    if start_year:
        # only stocks withing the desired time period
        df_concat = df_concat[start_year:] #from start_year

    old_shape = df_concat.shape

    # STOCKS SELECTION
    if last_traded:
        df_concat = df_concat.loc[:, df_concat.tail(30).notna().any()] #require to have at least one value in the last 30 days
    
    if min_non_na_fraction_col :
        # only stocks with many sufficiently many time points
        q = min_non_na_fraction_col * df_concat.shape[0]
        df_concat = df_concat.loc[:, df_concat.count() >= q]

    # DATES SELECTION
    if min_non_na_fraction_row:
        # only dates with data from most of the stocks
        threshold = min_non_na_fraction_row * df_concat.shape[1]
        df_concat = df_concat[df_concat.count(axis=1) > threshold] 
    
    # DATA MANIPULATION
    # calculate daily returns
    df_diff = df_concat.diff() #pct_change(fill_method=None) # confirmed that this should not be diff
    
    # remove outliers (eg. splits)    
    df_diff = df_diff.mask((df_concat.pct_change(fill_method=None).abs() > extreme_return_abs))
    
    # reduce local volatility
    if smooth_local_volatility:
        rolling_std = df_concat.rolling(window='30D', min_periods=10).std()
        df_concat = df_diff/rolling_std
    else:
        df_concat = df_diff

    # normalize    
    df_concat = (df_concat - df_concat.mean()) /df_concat.std()

    new_shape = df_concat.shape

    print(f'% of stocks remaining: {new_shape[1]/old_shape[1]:.2%}')
    print(f'% of dates remaining: {new_shape[0]/old_shape[0]:.2%}')
    print(f'Number of stocks: {new_shape[1]}')
    print(f'Number of dates: {new_shape[0]}')
    print(f'% of nans: {df_concat.isna().sum().sum()/df_concat.size:.2%}')
    
    return df_concat


def LI(df, tau_list, gaussianize_I=False):  # influence shifted forward by tau! 
    LI_tau = []

    I = df.mean(axis=1)
    if gaussianize_I:
        I = gaussianize(I)

    I2 = I**2

    for tau in tau_list:        

        I2_mean = I2.shift(periods=tau).mean() # I2_mean = I2.shift(periods=-tau).mean()

        corr_mean = (I.shift(periods=tau) * I2).mean()

        LI_tau.append(corr_mean/I2_mean)
    return pd.Series(LI_tau, index=tau_list, name='LI_tau')

def Lsigma(df, tau_list, gaussianize_I=False):
    # df = df.copy().dropna()
    Lsigma_tau = []

    I = df.mean(axis=1)
    if gaussianize_I:
        I = gaussianize(I)
    I2 = I**2

    for tau in tau_list:        

        I2_mean = I2.shift(periods=tau).mean() # I2_mean = I2.shift(periods=-tau).mean()

        corr_mean = (I.shift(periods=tau) * (df**2).mean(axis=1)).mean()

        Lsigma_tau.append(corr_mean/I2_mean)
    return pd.Series(Lsigma_tau, index=tau_list, name='Lsigma_tau')

def rho(df):
    # df = df.copy().dropna()

    N = df.count(axis=1) # for some dates there are nans
    I = df.mean(axis=1) 

    sigma2 = (df**2).mean(axis=1)  

    numerator = (I**2 * N**2) - (N * sigma2)
    denominator = N * (N - 1) * sigma2

    rho_t = numerator / denominator
    return rho_t


def Lrho(df, tau_list, gaussianize_I=False):
    # df = df.copy().dropna()

    Lrho_tau = []

    I = (df).mean(axis=1)
    if gaussianize_I:
        I = gaussianize(I)
    I2 = I**2

    rho_vals = rho(df)

    for tau in tau_list:    
        
        I2_mean = I2.shift(periods=tau).mean() # I2_mean = I2.shift(periods=-tau).mean()

        corr_mean = (I.shift(periods=tau) * rho_vals).mean()

        Lrho_tau.append(corr_mean/I2_mean)
    return pd.Series(Lrho_tau, index=tau_list, name='Lrho_tau')

def plot(df, fig, ax, gaussianize_I=False):
    tau_list = np.arange(1, 250, 1)
    rho_0 = rho(df).mean()
    sigma2_0 = (df**2).mean(axis=1).mean()
    LI_vals = LI(df, tau_list, gaussianize_I)
    Lsigma_vals = Lsigma(df, tau_list, gaussianize_I)
    Lrho_vals = Lrho(df, tau_list, gaussianize_I)
    LI_vals.plot(ax=ax[1], label=r'$L_I$', color='black', lw=0.8)
    (Lsigma_vals*rho_0).plot(ax=ax[0], label=r'$L_{\sigma}\rho_0$', color='red', lw=0.7)
    (Lrho_vals*sigma2_0).plot(ax=ax[0], label=r'$L_{\rho}\sigma_0^2$', color='blue', lw=0.7)
    ax[0].set_xlabel(r'$\tau$')
    ax[1].set_xlabel(r'$\tau$')

    fig.legend(loc='lower center')

    I2_mean = (((df).mean(axis=1))**2).mean()
    print(f'<I^2> = {I2_mean:.4f}')
    print(f'rho_0*sigma2_0 = {rho_0*sigma2_0:.4f}')
    print(f'rhomean = {rho_0:.4f}, rhostd = {rho(df).std():.4f}')


def gaussianize(I):
    """
    Gaussianize an array of index returns I.
    
    Parameters:
    - I: 1D numpy array of index returns.
    
    Returns:
    - I_G: Gaussianized version of I.
    """
    T = len(I)
    
    # Get the ranks (smallest = 1, largest = T)
    ranks = np.argsort(np.argsort(I.values)) + 1  # rank from 1 to T
    
    # Convert ranks to uniform quantiles (in (0,1) range)
    uniform_quantiles = ranks / (T + 1)  # avoid 0 and 1 to keep Φ⁻¹ defined
    
    # Apply the inverse CDF of the standard normal distribution (Gaussianize)
    I_G = norm.ppf(uniform_quantiles)
    
    return pd.Series(I_G, index=I.index)
