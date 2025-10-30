## ARIMA Model Evaluation Loop Index
A monthly evaluation of the ARIMA model class was successful, but it soon became apparent that something was amiss: the prediction array was one time step shorter than the array of true prices it was being compared against.

Debugging this was challenging due to the subtlety of the error. I examined the walk-forward validation process by manually tracking an iteration, which made the source of the problem clear.

In walk-forward validation, a model is trained on all data up to the end of the training set. It is then asked to make a forecast for a defined horizon (let's say `h` steps), starting from the very next time step. This next time step is, of course, the first time step of the validation set.

The issue was a classic **off-by-one indexing error** in how the forecast horizon `h` was being calculated. The number of steps to forecast (`h`) must be the length of the validation set.

This simple error meant that, for a 30-day validation set, my code was requesting only a 29-step forecast, resulting in an array mismatch. After correcting the horizon calculation, I conducted another evaluation, and the results were a success.


## ARIMA Model Handling the Gap in the Test Dataset

### The Gap Problem
The raw BTC dataset has a 1,160-row gap. I deleted the entire day's data (1,440 rows) for that date, as it constituted a negligible 0.0203% of the total dataset. However, this deletion poses a serious issue when the gap falls within the test set, as time series models expect continuous data.

### Naive Model Impact
A naive model ($forecast_t = actual_{t-1}$) is only minimally affected. The gap impacts just one forecast: the prediction for March 16th.

To make this prediction, the model would normally use the actual value from March 15th. Since that value is missing, the model must use the last available value (from March 14th). This effectively converts the 1-day forecast into a 2-day forecast (`Forecast_Mar_16 = Actual_Mar_14`). While this single point's error will be higher, the overall impact on the final MAPE and DA scores is negligible.

### ARIMA Model Impact
ARIMA models are a different story. They fundamentally assume that data arrives at regular, continuous intervals. A gap violates this.The model's core components—differencing ($d$), auto-regressive lags ($p$), and moving average errors ($q$)—all depend on a consistent time step. A two-day jump from the 14th to the 16th is not equivalent to two one-day steps and disrupts the patterns the model has learned. 

Unlike the naive model, this disruption can corrupt the model's internal state, negatively impacting the accuracy of several subsequent forecasts as it tries to recover. This makes the resulting performance metrics unreliable.

### Imputation Choice
For an ARIMA model, the best practice is to maintain a complete, regularly spaced DatetimeIndex in the test dataset. This means the missing day must be imputed.
- **Forward Fill (LOCF)**: This is simple, fills March 15th with the value from March 14th, and is causally correct (it uses no future information). Its drawback is creating an artificial flat line.
- **Linear Interpolation**: This creates a smoother, more realistic looking transition by drawing a line between the March 14th and March 16th values. However, it introduces significant look-ahead bias by using future data.

I chose linear **interpolation** because, although lookahead bias occurs, it is acceptable for repairing a small gap in a test dataset. Plus, it makes no sense at all that the prices remain the same for a day or a week.