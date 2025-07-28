from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def _save_plot(fig, filename):
    '''Saves a matplotlib figure to the "outputs" directory.'''
    output_dir = Path('outputs/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, bbox_inches='tight')
    print(f'Plot saved to "{output_dir / filename}".')
    plt.close(fig)

def plot_btc_price_distribution(df_series, filename_prefix):
    '''
    Plots the distribution of the closing price and its log-transformed version.

    Args:
        df_series (pd.DataFrame): The DataFrame containing the 'close' price column.
        filename_prefix (str): A prefix for the output plot filenames.
    '''
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(df_series['close'], bins=50, kde=True, ax=ax1)
    ax1.set_title('Distribution of BTC Daily Closing Price')
    ax1.set_xlabel('Price (USD)')
    ax1.set_ylabel('Frequency')
    _save_plot(fig1, f'btc_{filename_prefix}_distribution.png')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    log_prices = np.log1p(df_series['close'])
    sns.histplot(log_prices, bins=50, kde=True, ax=ax2, color='orange')
    ax2.set_title('Distribution of BTC Log-Transformed Daily Closing Price')
    ax2.set_xlabel('Log(1 + Price)')
    ax2.set_ylabel('Frequency')
    _save_plot(fig2, f'btc_log_{filename_prefix}_distribution.png')

def plot_btc_volume_distribution(df_series, filename_prefix):
    '''
    Plots the distribution of the trading volume and its log-transformed version.

    Args:
        df_series (pd.DataFrame): The DataFrame containing the 'volume' column.
        filename_prefix (str): A prefix for the output plot filenames.
    '''
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(df_series['volume'], bins=50, kde=True, ax=ax1)
    ax1.set_title('Distribution of BTC Daily Trading Volume')
    ax1.set_xlabel('Volume (BTC)')
    ax1.set_ylabel('Frequency')
    _save_plot(fig1, f'btc_{filename_prefix}_distribution.png')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    log_volumes = np.log1p(df_series['volume'])
    sns.histplot(log_volumes, bins=50, kde=True, ax=ax2, color='orange')
    ax2.set_title('Distribution of BTC Log-Transformed Daily Trading Volume')
    ax2.set_xlabel('Log(1 + Volume)')
    ax2.set_ylabel('Frequency')
    _save_plot(fig2, f'btc_log_{filename_prefix}_distribution.png')

def plot_btc_price_boxplot(df_series, filename='btc_price_boxplot.png'):
    '''
    Creates a boxplot for the closing price to visualize price distribution over time.

    Args:
        df_series (pd.DataFrame): The DataFrame containing OHLC columns.
    '''
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Create a year-month column for grouping.
    df = df_series.copy()
    df['year_month'] = df.index.to_period('M')
    
    # Create boxplots for each month.
    sns.boxplot(x='year_month', y='close', data=df, ax=ax)
    ax.set_title('Monthly Distribution of BTC Daily Closing Price')
    ax.set_xlabel('Month')
    ax.set_ylabel('Price (USD)')
    plt.xticks(rotation=90)
    _save_plot(fig, filename)

def plot_btc_volume_boxplot(df_series, filename='btc_volume_boxplot.png'):
    '''
    Creates a boxplot for the trading volume to visualize volume distribution over time.

    Args:
        df_series (pd.DataFrame): The DataFrame containing the 'volume' column.
    '''
    fig, ax = plt.subplots(figsize=(18, 8))

    df = df_series.copy()
    df['year_month'] = df.index.to_period('M')

    sns.boxplot(x='year_month', y='volume', data=df, ax=ax)
    ax.set_title('Monthly Distribution of BTC Daily Trading Volume')
    ax.set_xlabel('Month')
    ax.set_ylabel('Volume (BTC)')
    plt.xticks(rotation=90)
    _save_plot(fig, filename)

def plot_btc_price_trend(df_series, filename='btc_price_trend.png'):
    '''Plots the closing price over time.'''
    # Golden Cross(50-day > 200-day -> bullish signal) vs. Death Cross(50-day < 200-day -> bearish signal).    
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df_series.index, df_series['close'], label='Closing Price', alpha=0.4, color='steelblue')
    ax.plot(df_series['close'].rolling(window=50).mean(), 
            label='50-Day Rolling Avg', color='orange', linewidth=2)
    ax.plot(df_series['close'].rolling(window=200).mean(), 
            label='200-Day Rolling Avg', color='red', linewidth=2)
    ax.set_title('BTC Daily Closing Price Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)

def plot_btc_price_rollig_volatility(df_series, filename='btc_price_rolling_volatility.png'):
    '''Plots the closing price rolling volatility over time.'''
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df_series.index, df_series['rolling_volatility'], 
            label='30-Day Rolling Volatility', color='purple')
    ax.set_title('BTC Daily Closing Price Rolling Historical Volatility')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)

def plot_btc_volume_trend(df_series, filename='btc_volume_trend.png'):
    '''Plots the trading volume over time.'''
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df_series.index, df_series['volume'], label='Trading Volume', alpha=0.5, color='steelblue')
    ax.plot(df_series['volume'].rolling(window=50).mean(), 
            label='50-Day Rolling Avg', color='orange', linewidth=2)
    ax.plot(df_series['volume'].rolling(window=200).mean(), 
            label='200-Day Rolling Avg', color='red', linewidth=2)

    top_10_daily_volume_spikes = df_series.sort_values('volume', ascending=False).head(10)

    ax.scatter(
        top_10_daily_volume_spikes.index,
        top_10_daily_volume_spikes['volume'],
        color='red',
        s=100,
        edgecolor='black',
        label='Top 10 Volume Spikes',
        zorder=5
    )

    ax.set_title('BTC Daily Trading Volume Trend with Highlighted Spikes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume (BTC)')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)

def plot_btc_price_and_volume_corr(df_series, filename='btc_price_and_volume_corr.png'):
    '''Plots the correleation between the closing price and trading volume.'''
    # Bearish Divergence(prices reach a new high on lower volume than the previous high) 
    # Bullish Divergence(prices reach a new low on lower volume than the previous low)    
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(
        df_series['volume'],
        df_series['close'],
        color='steelblue',
        edgecolor='black',
    )

    ax.set_title('Correlation Between BTC Daily Closing Price and Trading Volume')
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Volume (BTC)')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)

def plot_btc_price_and_volume(df_series, filename='btc_price_and_volume.png'):
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    ax1.set_title('BTC Daily Closing Price')
    ax1.plot(df_series.index, df_series['close'], label='Closing Price', color='dodgerblue')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.set_title('BTC Daily Trading Volume')
    ax2.bar(df_series.index, df_series['volume'], label='Trading Volume', color='lightgray')

    # Add a moving average to the volume to see its trend more clearly.
    daily_volume_ra = df_series['volume'].rolling(window=30).mean()
    ax2.plot(daily_volume_ra.index, daily_volume_ra, label='30-Day Rolling Avg', color='darkorange')

    ax2.set_ylabel('Volume (BTC)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    _save_plot(fig, filename)

def plot_stationarity_analysis(df_series, col, filename='btc_price_stationarity_analysis.png'):
    '''Plots the rolling mean and rolling standard deviation of the closing price over time.'''
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df_series.index, df_series[col], label='Closing Price', alpha=0.4, color='steelblue')
    ax.plot(df_series[col].rolling(window=30).mean(), 
            label='30-Day Rolling Mean', color='orange', linewidth=2)
    ax.plot(df_series[col].rolling(window=30).std(), 
            label='30-Day Rolling Std Dev', color='red', linewidth=2)

    ax.set_title('Rolling Mean and Rolling Std Dev')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)    

def plot_autocorrelation(df_series, col, 
                         lags, title, filename_prefix):
    # Autocorreleation describes how a time series data point relates to its past values.
    # ACF and PACF plots are crucial for identifying the presence/strength of seasonality and for determining the parameters of time-series models.
    # ACF plots show the correlation of the time series with lags.
    # PACF plots show the correlation of the time series with lags but with the intervening correlations removed.
    ''' Plots the ACF and PACF for a given time series.'''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    series = df_series[col].dropna()

    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title(f'ACF ({title})')

    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title(f'PACF ({title})')

    plt.tight_layout()
    
    _save_plot(fig, f'{filename_prefix}_autocorrelation.png')

def plot_time_series_decomposition(df_series, col, model, 
                                   period, title, filename='btc_price_decomposition.png'):
    '''Performs and plots time-series decomposition (trend, seasonality and residuals).'''
    series = df_series[col].dropna()
    decomposition = seasonal_decompose(series, model=model, period=period)

    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    fig.supxlabel('Date')
    fig.suptitle(f'{title}', fontsize=14)
    for ax in fig.get_axes():
      ax.set_title('')
    fig.tight_layout()
    
    _save_plot(fig, filename)

def plot_btc_price_and_differencing(
    df_series, 
    lag,
    filename='btc_price_and_differencing.png'
):
    '''
    Visualizes the effect of lagging and differencing on the closing price.

    This plot helps in understanding how differencing, a common technique to
    make a time series stationary, transforms the data. It displays three plots:
    1. The original closing price vs. the price lagged by a specified period.
    2. The differenced closing price.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'close' price column.
        lag (int): The number of days to use for the lag and differencing.
        filename (str): The filename for the saved plot.
    '''    
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
    )
    fig.suptitle(f'{lag}-Day Lag and Differencing (BTC Daily)', fontsize=16, y=1.02)

    df_series['close'].plot(ax=ax1, label='Closing Price', grid=True)
    df_series['close'].shift(lag).plot(ax=ax1, label=f'{lag}-Day Lagged Price', grid=True, linestyle='--')
    ax1.set_title('Original Price vs. Lagged Price')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')

    df_series['close'].diff(lag).plot(ax=ax2, label='Differenced Price', grid=True, color='green')
    ax2.set_title(f'{lag}-Day Differenced Price')
    ax2.set_ylabel('Price Difference (USD)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 1])

    _save_plot(fig, filename)

def plot_residuals_analysis(df_results, col, lags):
    # Residuals should look like white noise.
    # That is, distribution of residuals should be a Gaussian with a zero mean.
    # Plot ACF/PACF graphs to check for autocorrelation.     
    if df_results is None:
        print('No best ARIMA model to analyze.')

    df_results = df_results[col]
    residuals = df_results['true_value'] - df_results['prediction']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Residual Analysis', fontsize=14)

    # Residuals over time.
    # Check for patterns or changing variance (heteroscedasticity).
    axes[0, 0].plot(residuals.index, residuals)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Error (USD)')
    axes[0, 0].grid(True)   

    # Histogram and KDE of Residuals.
    # Check if residuals are normally distributed around zero
    residuals.plot(kind='hist', bins=30, ax=axes[0, 1], density=True, label='Histogram')
    residuals.plot(kind='kde', ax=axes[0, 1], color='red', label='KDE')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].set_xlabel('Error (USD)')
    axes[0, 1].legend()

    # ACF Plot of Residuals.
    # Check for any remaining autocorrelation in the errors.
    plot_acf(residuals, lags=lags, ax=axes[1, 0])
    axes[1, 0].set_title(f'Autocorrelation for Residuals - First {lags} Lags')

    # PACF Plot of Residuals.
    # Check for remaining autocorrelation.
    plot_pacf(residuals, lags=lags, ax=axes[1, 1])
    axes[1, 1].set_title(f'Partial Autocorrelation for Residuals - First {lags} Lags')

    plt.tight_layout()
    plt.show()