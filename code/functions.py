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

def adjust_for_inflation(df, cpi_file_name):
    df_cpi = pd.read_csv(f'C:/physics_masters/data/consumer_price_index/{cpi_file_name}', parse_dates=['date'], index_col=0)

    cpi_daily = df_cpi.resample('D').ffill()

    # 2. Align CPI to the date range of price data
    cpi_daily = cpi_daily.reindex(df.index, method='ffill')

    # 3. Rebase to the earliest date in df_concat
    earliest_date = df.index.min()
    cpi_base_value = cpi_daily.loc[earliest_date, 'CPI']
    cpi_factor = cpi_base_value / cpi_daily['CPI']

    # 4. Adjust for inflation
    df_real = df.multiply(cpi_factor, axis=0)
    return df_real    


def concat_and_select(dataframes, min_non_na_fraction_col = 0.85, start_date = '2010', end_date='06-2024', extreme_return_abs = 2., min_non_na_fraction_row = 0.6, cpi_file=None):
    df_concat = pd.concat(dataframes, axis=1)
    df_concat.index = df_concat.index.astype(dataframes[0].index.dtype)
    df_concat.sort_index(inplace=True)

    # YEAR RANGE
    if start_date:
        # only stocks within the desired time period
        df_concat = df_concat[start_date:] #from start_date

    if end_date:
        # only stocks within the desired time period
        df_concat = df_concat[:end_date] #to end_date

    old_shape = df_concat.shape

    if cpi_file:
        df_concat = adjust_for_inflation(df_concat, cpi_file)


    # STOCKS SELECTION

    df_concat = df_concat.loc[:, df_concat.head(30).notna().any()] #require to have at least one value in the first 30 days
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
    if extreme_return_abs is not None:
        price_ratio = df_concat / df_concat.shift(1)
        df_concat[(price_ratio > extreme_return_abs) | (price_ratio < 1 / extreme_return_abs)] = np.nan

    new_shape = df_concat.shape

    print(f'% of stocks remaining: {new_shape[1]/old_shape[1]:.2%}')
    print(f'% of dates remaining: {new_shape[0]/old_shape[0]:.2%}')
    print(f'Number of stocks: {new_shape[1]}')
    print(f'Number of dates: {new_shape[0]}')

    return df_concat

def calculate_returns(df_concat, type='diff', smoothed=False):      #diff, log or pct_change    
    if type=='diff':
        df_diff = df_concat.diff()
        # df_diff = (df_diff - df_diff.mean()) /df_diff.std()
        # index_returns = df_diff.mean(axis=1)

    elif type == 'log':
        df_diff = np.log(df_concat).diff()

    elif type == 'pct_change':
        df_diff = df_concat.pct_change(fill_method=None)
  
    # reduce local volatility
    if smoothed:
        rolling_std = df_concat.rolling(window='30D', min_periods=10).std()
        df_diff = df_diff / rolling_std

    df_diff = (df_diff - df_diff.mean()) /df_diff.std()
  
    print(f'% of nans: {df_diff.isna().sum().sum()/df_concat.size:.2%}')
    
    #calculate index returns
    #index_series = df_concat.mean(axis=1)
    #index_returns = np.log(index_series/index_series.shift(1))
    #index_returns = (index_returns-index_returns.mean())/index_returns.std()

    index_returns = df_diff.mean(axis=1)

    return df_diff, index_returns


def LI(df_stocks, index_series, tau_list, gaussianize_I=False):  # influence shifted forward by tau! 
    LI_tau = []

    I = index_series

    if gaussianize_I:
        I = gaussianize(I)

    I2 = I**2

    for tau in tau_list:        

        I2_mean = I2.shift(periods=tau).mean() 

        corr_mean = (I.shift(periods=tau) * I2).mean()

        LI_tau.append(corr_mean/I2_mean)
    return pd.Series(LI_tau, index=tau_list, name='LI_tau')

def Lsigma(df_stocks, index_series, tau_list, gaussianize_I=False):
    # df = df.copy().dropna()
    Lsigma_tau = []

    I = index_series

    if gaussianize_I:
        I = gaussianize(I)
    I2 = I**2

    for tau in tau_list:        

        I2_mean = I2.shift(periods=tau).mean() # I2_mean = I2.shift(periods=-tau).mean()

        corr_mean = (I.shift(periods=tau) * (df_stocks**2).mean(axis=1)).mean()

        Lsigma_tau.append(corr_mean/I2_mean)
    return pd.Series(Lsigma_tau, index=tau_list, name='Lsigma_tau')

def rho(df):

    N = df.count(axis=1) # for some dates there are nans
    I = df.mean(axis=1) 

    sigma2 = (df**2).mean(axis=1)  

    numerator = (I**2 * N**2) - (N * sigma2)
    denominator = N * (N - 1) * sigma2

    rho_t = numerator / denominator
    return rho_t


def Lrho(df_stocks, index_series, tau_list, gaussianize_I=False):

    Lrho_tau = []

    I = index_series

    if gaussianize_I:
        I = gaussianize(I)
    I2 = I**2

    rho_vals = rho(df_stocks)

    for tau in tau_list:    
        
        I2_mean = I2.shift(periods=tau).mean() # I2_mean = I2.shift(periods=-tau).mean()

        corr_mean = (I.shift(periods=tau) * rho_vals).mean()

        Lrho_tau.append(corr_mean/I2_mean)
    return pd.Series(Lrho_tau, index=tau_list, name='Lrho_tau')

def calculate_correlation_functions(df_stocks, index_series, gaussianize_I=False):
    tau_list = np.arange(1, 250, 1)

    rho_0 = rho(df_stocks).mean()
    sigma2_0 = (df_stocks**2).mean(axis=1).mean()
    I2_mean = (index_series**2).mean()

    LI_vals = LI(df_stocks, index_series, tau_list, gaussianize_I)
    Lsigma_vals = Lsigma(df_stocks, index_series, tau_list, gaussianize_I)
    Lrho_vals = Lrho(df_stocks, index_series, tau_list, gaussianize_I)

    return tau_list, LI_vals, Lsigma_vals, Lrho_vals, sigma2_0, rho_0, I2_mean


def plot_correlation_functions(data, fig, ax):
    tau_list, LI_vals, Lsigma_vals, Lrho_vals, sigma2_0, rho_0, I2_mean = data
    LI_vals.plot(ax=ax[1], label=r'$L_I$', color='black', lw=0.8)
    (Lsigma_vals*rho_0).plot(ax=ax[0], label=r'$L_{\sigma}\rho_0$', color='red', lw=0.7)
    (Lrho_vals*sigma2_0).plot(ax=ax[0], label=r'$L_{\rho}\sigma_0^2$', color='blue', lw=0.7)
    ax[0].set_xlabel(r'$\tau$')
    ax[1].set_xlabel(r'$\tau$')
    ax[0].set_ylabel('Adjusted regression coefficients')
    ax[1].set_ylabel('Regression coefficients')

    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')

    print(f'<I^2> = {I2_mean:.4f}')
    print(f'rho_0*sigma2_0 = {rho_0*sigma2_0:.4f}')
    print(f'rho_0 = {rho_0:.4f}, sigma2_0 = {sigma2_0:.4f}')



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
