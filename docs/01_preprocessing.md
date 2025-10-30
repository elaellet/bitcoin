## Missing Timestamps
During the data integrity check, I discovered a large, continuous gap of 1,160 missing rows, spanning 19 hours and 20 minutes on March 15, 2025 (00:01:00 to 19:20:00). The data source notes that such errors can result from API downtime or other technical issues.

This left two options: **imputation or deletion**.

Simple imputation methods like forward fill or linear interpolation were not viable. While the price moved smoothly across the gap's boundaries, using these methods would mean fabricating 19 hours of data. This artificial data would inherently kill all volatility and introduce a pattern that never occurred. Since a model's predictive power relies on learning true patterns of price movement and volatility, this fabrication would degrade its performance.

Given that the missing data accounted for the vast majority of that day (over 80%), retaining the few remaining data points would have resulted in a fragmented, unrepresentative trading day.

To preserve the integrity of the dataset's patterns, I decided to remove the entire day's data. Therefore, all 1,440 rows for that date were deleted. This loss constitutes approximately 0.0203% of the total dataset, making it a negligible and justifiable adjustment.


## Handling Outliers

### Anomaly Detection Strategy
After handling missing timestamps and verifying data integrity (checking for negative prices, zero trading volumes, etc.), I began searching for anomalous data points. These data errors, such as large spikes or crashes, often exhibit a specific signature: a single point at a wildly different price that immediately reverts, often in conjunction with zero or abnormally low volume.

To find these, I chose two complementary metrics: **percentage change** (to analyze minute-to-minute inter-period changes) and **intra-minute spread** (to analyze volatility within a single minute).

### Setting Thresholds
I encountered a dilemma in setting thresholds for these metrics. The first approach, using standard deviations, is common but ill-suited for Bitcoin. Financial data does not follow a normal distribution; it exhibits **fat tails**, **skewness**, and **volatility clustering**.

A more robust approach is to use **percentiles** or **quantiles**, which are non-parametric (they don't assume a specific distribution). A subjective threshold (e.g., "any 20% change is an outlier") is less rigorous. I opted for the robust, statistical quantile approach. Since I was focused on the magnitude of extreme, one-sided events (spikes and crashes) rather than the spread of the central data, I chose to use high quantiles over the Interquartile Range (IQR).

### Data Preparation and Detection
To prepare the data, I engineered the percentage change and intra-minute spread features.
  1. **Gap Handling**: In a prior step, an entire day's worth of data (1,440 rows) was deleted. To prevent an artificial overnight spike in the percentage change calculation, I first calculated the time difference between consecutive rows. I set the `percentage_change` value to `NaN` wherever this gap exceeded one minute. This neutralizes artificial spikes and crashes that occur when a single day of data is deleted from the dataset.
  2. **Spread Calculation**: The `intra-minute_spread` was calculated as `(high - low) / open`.
  3. **Outlier Detection**: I applied a two-step process to identify outliers.
     - **Data-Driven**: I first identified candidates using data-driven quantiles. Any point with a price change (or spread) above the 99.99th percentile and a trading volume below the 5th percentile was flagged. This targets the classic low-volume spike error.
     - **Fixed Heuristic**: I then applied a second, fixed rule. Any price movement (percentage change) exceeding an absolute value of 0.75 (a 75% change in one minute) was designated an outlier regardless of volume, as such an event is a data error by definition.
This two-step process—using quantiles to target extreme data points and a simple, fixed threshold for the final exclusion—was chosen because it is more robust than relying solely on an arbitrary heuristic, while avoiding the complexity of building a fully data-driven exclusion model. 

### Imputation
Finally, I forward-filled the `NaN` values created by deleting the outliers. I chose this over linear interpolation for two reasons:
  1. The dataset's frequency is one minute. A price remaining steady for 1 minute (the assumption of forward fill) is plausible in the real world.
  2. Linear interpolation would create an artificial linear trend, fabricating a pattern that did not exist. Forward-fill (Last Observation Carried Forward) is a more conservative and appropriate choice for imputing small, discrete gaps.


## Log Transformation and Differencing

### The Problem with Raw Price Data
As part of my data exploration, I benchmarked a naive persistence forecast using MAE and MAPE at various horizons (hourly, daily, weekly, etc.). The results highlighted a fundamental problem: Bitcoin's extreme volatility, in which price fluctuations scale with the price level, is a classic example of **heteroscedasticity**. This makes raw prices an unstable and unreliable target for both modeling and evaluation.

### Log Transformation for Variance
The standard solution for this is a log transformation. This conversion addresses two issues simultaneously:
  1. **Stabilizes Variance**: It compresses the data's scale, making the variance more consistent and achieving homoscedasticity.
  2. **Linearizes Trends**: It converts exponential trends into linear ones, which models like ARIMA can capture more effectively.

Critically, using a log-transform also improves the error metric. Minimizing the Mean Absolute Error (MAE) on log-prices (`abs(log(P_true) - log(P_pred))`) is a robust proxy for minimizing the Mean Absolute Percentage Error (MAPE). This is because the difference in logs is a close approximation of a percentage change.

This log-MAE is scale-independent, making it appropriate for assets with time-varying volatility. It also avoids the divide-by-zero error that MAPE suffers from, striking an outstanding balance between mathematical rigor and practical interpretation.

### Differencing for Stationarity
However, even log-transformed prices are non-stationary—they still possess a trend and do not revert to a mean. Differencing is the standard technique to correct this. By subtracting the previous observation from the current one, differencing removes the trend and seasonality, making the series stationary.

### Log Returns
Combining two techniques achieves two objectives.
  1. **Log Transformation**: Stabilizes the variance
  2. **Differencing**: Removes the trend and achieves stationarity.

The result of this two-step process, `log(P_t) - log(P_{t-1})`, is the **log return**. This series is both stationary and homoscedastic, making it the ideal input for building a robust time series model.


## Box-Cox vs. Log Transformation 
When computing the MAPE for an optimal ARIMA model on a daily resampled dataset, I found the results disappointing; the forecast was only marginally better than a naive prediction. This suggested the model was struggling with the data's properties, and I suspected a Box-Cox transformation could help.

I ran the `boxcox()` function on the cleaned, raw dataset to find the optimal transformation parameter, which yielded a $\lambda$ (lambda) value of 0.142.

Since a $\lambda$ value of 0 corresponds precisely to the natural log transformation, my result of 0.142 confirmed that a log transform was already a near-optimal choice for stabilizing variance and achieving near-normality. This narrowed the choice: should I use the statistically optimal $\lambda = 0.142$ or the practically simpler log transformation ($\lambda = 0$)?

### The Practical Problem with Box-Cox
I ultimately favored the log transformation. While a Box-Cox with $\lambda = 0.142$ might be statistically superior, its output lacks the intuitive financial meaning of a log-transformed price. When differenced, the log transformation produces log returns, a fundamental and interpretable concept in quantitative finance.

Despite this preference, I ran an ARIMA grid search on the Box-Cox-transformed data to be certain. The results were abysmal. Upon debugging, I discovered the problem:
The optimal $\lambda$ calculated on the training set was slightly different from the $\lambda$ that would be optimal for the validation set. This discrepancy is critical. During backtesting, using the training set's $\lambda$ to transform the data is a must, but using its inverse to convert the validation forecasts back to the original price scale for evaluation is also required. Using a mismatched $\lambda$ for this inverse transformation produced wildly incorrect evaluation metrics.

Using the overall $\lambda$ of 0.142 for all sets would pose a significant data-snooping risk, as that value was partially derived from the validation and test sets.

Ultimately, I reverted to the log transformation. It provides a reliable, fixed ($\lambda=0$) method that entirely avoids the ambiguity and data leakage associated with calculating a representative $\lambda$ value across different data splits.


## Timeframe Selection
After developing initial versions of the naive and ARIMA models, I faced a significant dilemma: for a single forecast horizon (e.g., 1 month), there could be multiple "best" models, each trained on a different data frequency (e.g., hourly, daily, or weekly). This problem is compounded by the fact that comprehensive hyperparameter searches, especially for ARIMA and LSTM/GRU models, are incredibly time-consuming without access to powerful hardware like GPUs.

This realization shifted my focus from finding a single optimal model to defining a logical framework for selecting the appropriate data frequency for each task. 

The strategy is to match the data frequency to the forecast horizon.

  1. **Monthly Forecasts $\rightarrow$ Daily Data**: A daily dataset provides the ideal balance for a 30-day forecast. It captures significant weekly and monthly trends without being overwhelmed by the high-frequency noise inherent in hourly or minutely data. For this horizon, daily data is computationally efficient and focuses the model on the most relevant signals.
  2. **Yearly Forecasts $\rightarrow$ Weekly Data**: When forecasting a full year (52 weeks), using a daily dataset (365 points) can be noisy and computationally intensive. Resampling the data to a weekly frequency effectively smooths out the short-term daily fluctuations. This encourages the model to learn more stable, long-term trends and seasonal patterns, providing a practical, much faster approach to long-horizon forecasting.


## Directional Accuracy vs. MAPE
The project's initial goal was to predict the exact Bitcoin price, but evaluating the first models revealed a new priority: predicting the direction of the price movement. For trading, Directional Accuracy (DA) is often more valuable than magnitude-based error metrics like MAPE.

This created a comparison challenge. For instance, one model might have a high MAPE (worse magnitude error) but also a high DA (better direction), while another has the opposite. Averaging MAPE and DA is illogical, as they measure fundamentally different concepts (magnitude vs. direction) on different scales.

In a trading context, knowing the direction of the next move is far more critical than knowing its exact price. A model with high DA, even with a mediocre MAPE, is more likely to inform a profitable strategy than one with low MAPE and poor DA. This is because a correct directional forecast allows for a profitable trade, even if the predicted magnitude is slightly off.

### Hierarchical Model Selection
Thus, I decided to use a **hierarchical decision-making process** for model selection:
- 1. **Baseline Filter**: First, a model must perform better than the naive baseline on all key metrics (e.g., MAPE and DA).
- 2. **Primary Metric (DA)**: From the filtered set, I select the models with the highest Directional Accuracy (DA).
- 3. **Secondary Metric (MAPE)**: If multiple models tie for the highest DA, I use the lowest MAPE as the tiebreaker to select the single best model.


## Split Function vs. Index-Based Slicing

### The Split-Then-Transform Pitfall
Calculating log returns after splitting the data into training, validation, and test sets caused a serious issue. The `.diff()` operation introduced a `NaN` in the first row of each dataset. The subsequent `.dropna()` call then removed this row from each split.

This created a subtle one-timestep misalignment between the datasets, which corrupted the comparative index when evaluating LSTM/GRU forecasts against ARIMA forecasts.

I realized the correct approach is to **transform first, then split**. This involves concatenating the raw datasets, calculating log returns on the complete time series, handling the single `NaN` value at the very beginning, and then splitting the transformed data back into training, validation, and test sets.

The original percentage-based split function was the source of this discrepancy. Percentage-based splitting (e.g., 80% train, 10% validation) is unreliable for time series. It's an approximation and cannot guarantee that the splits occur at the exact identical timestamps across different model pipelines.

This problem is exacerbated by transformations: after the `log().diff().dropna()` step, the dataset is one row shorter, making percentage splits even more likely to misalign with the original, intended split dates.

The split_btc_dataset function, which used percentage-based splitting, was the source of the index discrepancy. The split_btc_dataset function divides a dataset into proportional chunks (e.g., 80% for training, 10% for validation). While it works perfectly with a single, clean dataset, it becomes unreliable after transformation for two reasons.

Index-based slicing is the precise and robust solution. This method involves saving the exact timestamps (e.g., `train_end_date`, `valid_end_date`) from the original raw data before any preprocessing. After the combined dataset is transformed and the initial NaN is dropped, the `.loc[]` method uses these saved timestamps to slice the final dataset with surgical precision.

This method also prevents data overlap. By defining the slices with explicit start and end timestamps (e.g., train_df = transformed_df.loc[`:train_end_date`] and valid_df = transformed_df.loc[`valid_start_date:valid_end_date`]), we ensure the last day of the training data is not accidentally included as the first day of the validation data.

### Handling Internal `NaN`s
A final complication is the risk of `NaN`s within the dataset. This can occur if a price drops to 0, since `log(0) =- inf`, and the subsequent `.diff()` operation can result in `NaN` or `inf`.

To preserve the continuity and regular time-step interval of the series, these internal `NaN`s must be imputed, not dropped. Dropping them would destroy the dataset's temporal structure. Since these events are rare, a simple forward-fill is an effective method to handle these minor internal gaps.