import itertools
import numpy as np
from epftoolbox.evaluation import DM
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from .utils import print_header

def display_descriptive_statistics(series_ds, series_name):
    '''
    Prints a summary of the dataset's structure and descriptive statistics.

    Args:
        series_ds (pd.DataFrame): The dataset to analyze.
        series_name (str): A descriptive name for the dataset for printing.
    '''
    print_header(f'Descriptive Statistics: {series_name}')
    print('--- Dataset Info ---')
    series_ds.info()

    print('\n--- Statistical Summary ---')
    # Using .to_string() to ensure all columns are displayed.
    print(series_ds.describe().to_string())

def run_adf_test(series_ds, col, series_name): 
    '''
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

    The ADF test is a statistical test for determining if a time series is stationary,
    meaning its statistical properties (like mean and variance) do not change over time.
    A p-value below a certain threshold (typically 0.05) suggests that the null hypothesis
    can be rejected (that the series is non-stationary).

    Args:
        series_ds (pd.DataFrame): The dataset containing the time series data.
        col (str): The name of the column to test for stationarity.
        series_name (str): A descriptive name for the time series for printing.
    '''
    print_header(f'ADF Test: {series_name}')

    # Drop NaN values to ensure the test runs correctly.
    series = series_ds[col].dropna()
    result = adfuller(series)

    print(f'ADF Statistics: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
      print(f'\t{key}:{value:.4f}')

    if result[1] <= 0.05:
        print('Conclusion: The p-value is less than or equal to 0.05. The data is likely stationary and seasonal.\n')
    else:
        print('Conclusion: The p-value is greater than 0.05. The data is likely non-stationary and non-seasonal.\n')
        
def run_ljung_box_test(comp, col, lags, model):
    '''
    Performs the Ljung-Box test to check for autocorrelation in residuals.

    The null hypothesis (H₀) is that the residuals are independently distributed
    (i.e., correlations in the population from which the sample is taken are all zero).

    Args:
        comp (pd.DataFrame): The DataFrame with the actual vs. predicted values.
        col (str): The name of the specific model or column to analyze.
        lags (int): The number of lags to test.
        model (str): The name of the model for display purposes.
    '''
    print_header(f'Ljung-Box Test: {model}')

    results = comp[col]
    residuals = results['true'] - results['pred']

    lb_test_results = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    p_value = lb_test_results['lb_pvalue'].iloc[0]

    print(f'Lags tested: {lags}')
    print(f'p-value: {p_value:.4f}')

    alpha = 0.05
    if p_value < alpha:
        print(f'Result: Reject the null hypothesis (p < {alpha}).')
        print('The residuals show evidence of autocorrelation.')
    else:
        print(f'Result: Fail to reject the null hypothesis (p >= {alpha}).')
        print('The residuals appear to be independently distributed.')

def run_dm_test(true_values, model_1_preds, model_2_preds,
                model_1_name='Model 1', model_2_name='Model 2',
                alpha=0.05, norm=2):
    '''
    Performs and interprets a Diebold-Mariano test between two models.

    Args:
        true_values (np.array): The true observed values.
        model_1_preds (np.array): Predictions from the first model.
        model_2_preds (np.array): Predictions from the second model.
        model_1_name (str): The name of the first model.
        model_2_name (str): The name of the second model.
        alpha (float): The significance level for the test.
        norm (int): The norm for the loss function (2 for squared error, 1 for absolute error).
    '''
    print(f'--- Diebold-Mariano Test: {model_1_name} vs. {model_2_name} ---')

    # The epftoolbox.DM function requires a 2D array of shape (n_days, n_prices_day)
    # where n_prices_day > 1. Reshape the 1D data to satisfy this.
    tv_flat = np.asarray(true_values).flatten()
    p1_flat = np.asarray(model_1_preds).flatten()
    p2_flat = np.asarray(model_2_preds).flatten()

    n_obs = len(tv_flat)

    # Check if n_obs is even. If not, drop one value.
    if n_obs % 2 != 0:
        print(f'Warning: DM test requires even observations. Dropping first value (obs count: {n_obs} -> {n_obs-1}).')
        tv_flat = tv_flat[1:]
        p1_flat = p1_flat[1:]
        p2_flat = p2_flat[1:]
        n_obs -= 1

    new_shape = (n_obs // 2, 2)
    tv_reshaped = tv_flat.reshape(new_shape)
    p1_reshaped = p1_flat.reshape(new_shape)
    p2_reshaped = p2_flat.reshape(new_shape)

    print(f'- Reshaping data from ({n_obs},) to {new_shape} for DM test compatibility.')

    p_value = DM(tv_reshaped, p1_reshaped, p2_reshaped, norm, 'multivariate')

    print(f'- p-value: {p_value:.4f}')
    
    if p_value < alpha:
        print(f'Conclusion: The difference is statistically significant (p < {alpha}).')
        loss1 = np.mean(np.abs(true_values - model_1_preds)**norm)
        loss2 = np.mean(np.abs(true_values - model_2_preds)**norm)

        if loss1 < loss2:
            print(f'- Result: {model_1_name} is significantly better than {model_2_name}.')
        else:
            print(f'- Result: {model_2_name} is significantly better than {model_1_name}.')
    else:
        print(f'- Conclusion: The difference is not statistically significant (p >= {alpha}).')
        print('- Result: Neither model can be considered superior to the other.')

def run_all_dm_tests(true_values, model_preds, alpha=0.05, norm=2):
    '''
    Orchestrates the Diebold-Mariano test for all unique pairs of models.

    Args:
        true_values (np.array): The true observed values.
        model_preds (dict): A dictionary where keys are model names (str)
            and values are their prediction arrays (np.array).
            Example:
            {
                'Naive': preds_naive,
                'ARIMA': preds_arima,
                'GRU': presd_gru,
                ...
            }
        alpha (float): The significance level for the test.
        norm (int): The norm for the loss function (2 for squared error, 1 for absolute error).
    '''
    print_header('Running All Model Pairwise Diebold-Mariano Tests')

    model_names = list(model_preds.keys())

    if len(model_names) < 2:
        print('Warning: Need at least two models to perform a DM test.')
        return
    
    for (model_1_name, model_2_name) in itertools.combinations(model_names, 2):
        print(f'\n' + '='*80 + '\n')

        model_1_preds = model_preds[model_1_name]
        model_2_preds = model_preds[model_2_name]

        run_dm_test(true_values, 
                    model_1_preds, model_2_preds, 
                    model_1_name, model_2_name, 
                    alpha, norm)

    print('\n' + '='*80 + '\n')
    print_header('All pairwise Diebold-Mariano tests complete.')