from statsmodels.tsa.stattools import adfuller

from .utils import print_header

def display_descriptive_statistics(df_series, series_name):
    '''
    Prints a summary of the DataFrame's structure and descriptive statistics.

    Args:
        df_series (pd.DataFrame): The DataFrame to analyze.
        series_name (str): A descriptive name for the dataset for printing.
    '''
    print_header(f'Descriptive Statistics: {series_name}')
    print('--- DataFrame Info ---')
    df_series.info()

    print('\n--- Statistical Summary ---')
    # Using .to_string() to ensure all columns are displayed.
    print(df_series.describe().to_string())

def run_adf_test(df_series, col, series_name): 
    '''
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

    The ADF test is a statistical test for determining if a time series is stationary,
    meaning its statistical properties (like mean and variance) do not change over time.
    A p-value below a certain threshold (typically 0.05) suggests that the null hypothesis
    can be rejected (that the series is non-stationary).

    Args:
        df_series (pd.DataFrame): The DataFrame containing the time series data.
        col (str): The name of the column to test for stationarity.
        series_name (str, optional): A descriptive name for the time series for printing.
    '''
    print_header(f'ADF Test: {series_name}')

    # Drop NaN values to ensure the test runs correctly.
    series = df_series[col].dropna()
    result = adfuller(series)

    print(f'ADF Statistics: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
      print(f'\t{key}:{value:.4f}')

    if result[1] <= 0.05:
        print('Conclusion: The p-value is less than or equal to 0.05. The data is likely stationary and seasonal.')
    else:
        print('Conclusion: The p-value is greater than 0.05. The data is likely non-stationary and non-seasonal.')