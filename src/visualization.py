from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def _save_plot(fig, filename):
    '''
    Saves a matplotlib figure to the "outputs" directory.
    
    Args:
        fig: The matplotlib Figure object to be saved.
        filename (str): A filename for the output plot.

    '''
    output_dir = Path('outputs/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, bbox_inches='tight')
    print(f'Plot saved to \'{output_dir / filename}\'.')
    plt.close(fig)

def plot_btc_price_distribution(series_ds, filename_prefix):
    '''
    Plots the distribution of the closing price and its log-transformed version.

    Args:
        series_ds (pd.DataFrame): The dataset containing the 'close' price column.
        filename_prefix (str): A prefix for the output plot filenames.
    '''
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(series_ds['close'], bins=50, kde=True, ax=ax1)
    ax1.set_title('Distribution of BTC Daily Closing Price')
    ax1.set_xlabel('Price (USD)')
    ax1.set_ylabel('Frequency')
    _save_plot(fig1, f'btc_{filename_prefix}_distribution.png')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    log_prices = np.log1p(series_ds['close'])
    sns.histplot(log_prices, bins=50, kde=True, ax=ax2, color='orange')
    ax2.set_title('Distribution of BTC Log-Transformed Daily Closing Price')
    ax2.set_xlabel('Log(1 + Price)')
    ax2.set_ylabel('Frequency')
    _save_plot(fig2, f'btc_log_{filename_prefix}_distribution.png')

def plot_btc_volume_distribution(series_ds, filename_prefix):
    '''
    Plots the distribution of the trading volume and its log-transformed version.

    Args:
        series_ds (pd.DataFrame): The dataset containing the 'volume' column.
        filename_prefix (str): A prefix for the output plot filenames.
    '''
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(series_ds['volume'], bins=50, kde=True, ax=ax1)
    ax1.set_title('Distribution of BTC Daily Trading Volume')
    ax1.set_xlabel('Volume (BTC)')
    ax1.set_ylabel('Frequency')
    _save_plot(fig1, f'btc_{filename_prefix}_distribution.png')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    log_volumes = np.log1p(series_ds['volume'])
    sns.histplot(log_volumes, bins=50, kde=True, ax=ax2, color='orange')
    ax2.set_title('Distribution of BTC Log-Transformed Daily Trading Volume')
    ax2.set_xlabel('Log(1 + Volume)')
    ax2.set_ylabel('Frequency')
    _save_plot(fig2, f'btc_log_{filename_prefix}_distribution.png')

def plot_btc_price_boxplot(series_ds, filename='btc_price_boxplot.png'):
    '''
    Creates a boxplot for the closing price to visualize price distribution over time.

    Args:
        series_ds (pd.DataFrame): The dataset containing the OHLC columns.
        filename (str): A filename for the output plot.
    '''
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Create a year-month column for grouping.
    df = series_ds.copy()
    df['year_month'] = df.index.to_period('M')
    
    # Create boxplots for each month.
    sns.boxplot(x='year_month', y='close', data=df, ax=ax)
    ax.set_title('Monthly Distribution of BTC Daily Closing Price')
    ax.set_xlabel('Month')
    ax.set_ylabel('Price (USD)')
    plt.xticks(rotation=90)
    _save_plot(fig, filename)

def plot_btc_volume_boxplot(series_ds, filename='btc_volume_boxplot.png'):
    '''
    Creates a boxplot for the trading volume to visualize volume distribution over time.

    Args:
        series_ds (pd.DataFrame): The dataset containing the 'volume' column.
        filename (str): A filename for the output plot.
    '''
    fig, ax = plt.subplots(figsize=(18, 8))

    df = series_ds.copy()
    df['year_month'] = df.index.to_period('M')

    sns.boxplot(x='year_month', y='volume', data=df, ax=ax)
    ax.set_title('Monthly Distribution of BTC Daily Trading Volume')
    ax.set_xlabel('Month')
    ax.set_ylabel('Volume (BTC)')
    plt.xticks(rotation=90)
    _save_plot(fig, filename)

def plot_btc_price_trend(series_ds, filename='btc_price_trend.png'):
    '''
    Plots the closing price over time.

    Args:
        series_ds (pd.DataFrame): The dataset containing the 'close' price column.
        filename (str): A filename for the output plot.
    '''
    # Golden Cross(50-day > 200-day -> bullish signal) vs. Death Cross(50-day < 200-day -> bearish signal).
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(series_ds.index, series_ds['close'], label='Closing Price', alpha=0.4, color='steelblue')
    ax.plot(series_ds['close'].rolling(window=50).mean(), 
            label='50-Day Rolling Avg', color='orange', linewidth=2)
    ax.plot(series_ds['close'].rolling(window=200).mean(), 
            label='200-Day Rolling Avg', color='red', linewidth=2)
    ax.set_title('BTC Daily Closing Price Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)

def plot_btc_price_rollig_volatility(series_ds, filename='btc_price_rolling_volatility.png'):
    '''
    Plots the closing price rolling volatility over time.
    
    Args:
        series_ds (pd.DataFrame): The dataset containing the 'close' price column.
        filename (str): A filename for the output plot.    
    '''
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(series_ds.index, series_ds['rolling_volatility'], 
            label='30-Day Rolling Volatility', color='purple')
    ax.set_title('BTC Daily Closing Price Rolling Historical Volatility')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)

def plot_btc_volume_trend(series_ds, filename='btc_volume_trend.png'):
    '''
    Plots the trading volume over time.
    
    Args:
        series_ds (pd.DataFrame): The dataset containing the 'volume' column.
        filename (str): A filename for the output plot.
    '''
    # Uptrend: A rising price trend accompanied by a rising VMA confirms the uptrend.
    # Downtrend: A falling price on a rising VMA confirms the downtrend.
    # Volume Spike: When daily volume spike is significantly above the VMA, it signals a major market event (e.g., capitulation, panic selling, or climactic buying) 
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(series_ds.index, series_ds['volume'], label='Trading Volume', alpha=0.5, color='steelblue')
    ax.plot(series_ds['volume'].rolling(window=50).mean(), 
            label='50-Day Rolling Avg', color='orange', linewidth=2)
    ax.plot(series_ds['volume'].rolling(window=200).mean(), 
            label='200-Day Rolling Avg', color='red', linewidth=2)

    top_10_daily_volume_spikes = series_ds.sort_values('volume', ascending=False).head(10)

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

def plot_btc_price_and_volume_corr(series_ds, filename='btc_price_and_volume_corr.png'):
    '''
    Plots the correleation between the closing price and trading volume.
    
    Args:
        series_ds (pd.DataFrame): The dataset containing the 'close' price and 'volume' column.
        filename (str): A filename for the output plot.    
    '''
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(
        series_ds['volume'],
        series_ds['close'],
        color='steelblue',
        edgecolor='black',
    )

    ax.set_title('Correlation Between BTC Daily Closing Price and Trading Volume')
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Volume (BTC)')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)

def plot_btc_price_and_volume(series_ds, filename='btc_price_and_volume.png'):
    '''
    Plots the relationship between the closing price and trading volume.
    
    Args:
        series_ds (pd.DataFrame): The dataset containing the 'close' price and 'volume' column.
        filename (str): A filename for the output plot.
    '''
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    ax1.set_title('BTC Daily Closing Price')
    ax1.plot(series_ds.index, series_ds['close'], label='Closing Price', color='dodgerblue')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.set_title('BTC Daily Trading Volume')
    ax2.bar(series_ds.index, series_ds['volume'], label='Trading Volume', color='lightgray')

    # Add a moving average to the volume to see its trend more clearly.
    daily_volume_ra = series_ds['volume'].rolling(window=30).mean()
    ax2.plot(daily_volume_ra.index, daily_volume_ra, label='30-Day Rolling Avg', color='darkorange')

    ax2.set_ylabel('Volume (BTC)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    _save_plot(fig, filename)

def plot_stationarity_analysis(series_ds, col, filename='btc_price_stationarity_analysis.png'):
    '''
    Plots the rolling mean and rolling standard deviation of the closing price over time.

    Args:
        series_ds (pd.DataFrame): The dataset containing the 'close' price and 'volume' column.
        col (str): The name of the target column.
        filename (str): A filename for the output plot.  
    '''
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(series_ds.index, series_ds[col], label='Closing Price', alpha=0.4, color='steelblue')
    ax.plot(series_ds[col].rolling(window=30).mean(), 
            label='30-Day Rolling Mean', color='orange', linewidth=2)
    ax.plot(series_ds[col].rolling(window=30).std(), 
            label='30-Day Rolling Std Dev', color='red', linewidth=2)

    ax.set_title('Rolling Mean and Rolling Std Dev')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left')
    ax.grid(True)

    _save_plot(fig, filename)

def plot_autocorrelation(series_ds, col, 
                         lags, title, filename_prefix):
    # ACF and PACF plots are crucial for identifying the presence/strength of seasonality and for determining the parameters of time-series models.
    # ACF plots show the correlation of the time series with lags.
    # PACF plots show the correlation of the time series with lags, but with the intervening correlations removed.
    ''' 
    Plots the ACF and PACF for a given time series.
    
    Args:
        series_ds (pd.DataFrame): The dataset containing the 'close' price and 'volume' column.
        col (str): The name of the target column.
        title (str): The title of the plots.
        filename_prefix (str): A prefix for the output plot filenames.
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    series = series_ds[col].dropna()

    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title(f'ACF ({title})')

    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title(f'PACF ({title})')

    plt.tight_layout()
    _save_plot(fig, f'{filename_prefix}_autocorrelation.png')

def plot_time_series_decomposition(series_ds, col, model, 
                                   period, title, filename='btc_price_decomposition.png'):
    '''
    Performs and plots time-series decomposition (trend, seasonality and residuals).

    Args:
        series_ds (pd.DataFrame): The dataset containing the 'close' price and 'volume' column.
        col (str): The name of the target column.
        title (str): The title of the plots.
        model (str): The type of seasonal component. 
        period (int): The period of the series.
        filename (str): A filename for the output plot.
    '''
    series = series_ds[col].dropna()
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
    series_ds, 
    lag,
    time_unit,
    filename='btc_price_and_differencing.png'
):
    '''
    Visualizes the effect of lagging and differencing on the closing price.

    This plot helps in understanding how differencing, a common technique to
    make a time series stationary, transforms the data. It displays three plots:
    1. The original closing price vs. the price lagged by a specified period.
    2. The differenced closing price.

    Args:
        df (pd.DataFrame): The dataset containing the 'close' price column.
        lag (int): The number of days to use for the lag and differencing.
        time_unit (str): The time unit for the forecast period.
        filename (str): A filename for the saved plot.
    '''    
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
    )
    fig.suptitle(f'{lag}-{time_unit} Lag and Differencing (BTC {time_unit})', fontsize=16, y=1.02)

    series_ds['close'].plot(ax=ax1, label='Closing Price', grid=True)
    series_ds['close'].shift(lag).plot(ax=ax1, label=f'{lag}-Day Lagged Price', grid=True, linestyle='--')
    ax1.set_title('Original Price vs. Lagged Price')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')

    series_ds['close'].diff(lag).plot(ax=ax2, label='Differenced Price', grid=True, color='green')
    ax2.set_title(f'{lag}-Day Differenced Price')
    ax2.set_ylabel('Price Difference (USD)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 1])
    _save_plot(fig, filename)

def plot_residuals_analysis(comp, col, lags, 
                            model, time_unit):
    '''
    Generates and displays a 2x2 grid of plots for residual analysis.

    This function assesses the performance of a time series model by visualizing
    its residuals (the difference between true and predicted values). It creates five key plots:
    1. Residuals over time to check for patterns or non-constant variance.
    2. A histogram and KDE plot to check if residuals are normally distributed.
    3. A Q-Q plot to visually check if residuals follow a normal distribution.
    4. An Autocorrelation Function (ACF) plot to detect remaining correlation.
    5. A Partial Autocorrelation (PACF) plot to detect remaining correlation.

    Args:
        comp (pd.DataFrame): The DataFrame with the actual vs. predicted values.
        col (str): The name of the specific model or column to analyze.
        lags (int): The number of lags to show in the ACF and PACF plots.
        model (str): The name of the model being analyzed.
        time_unit (str): The time unit for the forecast period.
    '''    
    # Plot ACF/PACF graphs to check for autocorrelation. 
    if comp is None or col not in comp:
        print(f'No data for {col} in the results to analyze.')

    results = comp[col]
    residuals = results['true'] - results['pred']

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(f'Residual Analysis for {model}', fontsize=16)

    # Residuals over time.
    axes[0, 0].plot(residuals.index, residuals)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Error (USD)')
    axes[0, 0].grid(True)   

    # Histogram and KDE of Residuals.
    residuals.plot(kind='hist', bins=30, ax=axes[0, 1], density=True, label='Histogram')
    residuals.plot(kind='kde', ax=axes[0, 1], color='red', label='KDE')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].set_xlabel('Error (USD)')
    axes[0, 1].legend()

    # Q-Q Plot of Residuals
    sm.qqplot(residuals, line='s', ax=axes[1, 0])
    axes[1, 0].set_title('Quantile-Quantile (Q-Q) Plot')

    # ACF Plot of Residuals.
    plot_acf(residuals, lags=lags, ax=axes[1, 1])
    axes[1, 1].set_title(f'Autocorrelation (ACF) - First {lags} Lags')

    # PACF Plot of Residuals.
    plot_pacf(residuals, lags=lags, ax=axes[2, 0])
    axes[2, 0].set_title(f'Partial Autocorrelation (PACF) - First {lags} Lags')

    fig.delaxes(axes[2, 1])

    filename = f'{model.lower()}_residuals_analysis_{time_unit.lower()}'
    plt.tight_layout()
    _save_plot(fig, filename)
