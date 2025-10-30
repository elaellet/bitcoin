## Backtesting
While implementing the **Diebold-Mariano (DM)** test, I confronted a key requirement: do the forecasts from competing models need to be generated using the same strategy? The answer is unequivocally yes. For a valid and fair comparison, all models must be evaluated using the exact same backtesting strategy. The DM test requires a true apples-to-apples comparison of the resulting forecast errors.

### Walk-Forward Validation
**Backtesting** is the time series equivalent of cross-validation, evaluating a model's performance on historical data. Crucially, it must strictly preserve the temporal order of the data to prevent peeking into the future (**lookahead bias**). This ensures the model is trained only on data that precedes the data it is tested on.

The key difference between backtesting strategies lies in how the training and test windows are moved and whether the model is refit for each new period. Walk-forward validation is the rigorous methodology that respects this arrow of time, making it essential for a fair comparison.

### Windowing and Refitting
This requirement highlighted a major inconsistency in my project: my existing ARIMA model used an expanding window with intermittent refits, while my LSTM/GRU model employed a rolling window without refits. To compare them fairly, this had to be standardized.

At any given time $t$ for making a forecast, both Model A and Model B must have access to the exact same set of historical information: same training data, same refit frequency, and same test data points. I decided to use rolling-window walk-forward validation with no refit across all models for the following reasons:
  1. **Computational Cost**: Retraining an RNN (i.e., calling the `fit()` method again) at every step of a walk-forward validation is computationally intensive and infeasible in practice. While statistical models like ARIMA often have efficient refit or update methods optimized for new data, neural networks generally require a more expensive retraining process.
  2. **Concept Drift**: Assets like Bitcoin are susceptible to concept drift—the market dynamics from a few years ago are likely different from the dynamics of the last six months. A rolling window allows a model to adapt by focusing only on the most recent, relevant data.
  3. **Model Sensitivity**: LSTMs and GRUs can be sensitive to the input sequence length. They may struggle to learn from very long sequences (e.g., 100+ time steps), though 1D CNNs can help by shortening the input. A fixed, rolling window size ensures a consistent input length at each step.

### Prediction Method
A critical problem remained: the choice of prediction strategy.
- **Direct Forecasting**: Makes a direct prediction of the price at the forecast horizon.
- **Recursive Forecasting**: Predicts a value just one step ahead and generates a forecast at the final horizon by feeding its own predictions back into itself as input.

Just like the windowing method, all models must use the same prediction method. Fortunately, I found that both the ARIMA and the LSTM/GRU code I had implemented already used direct forecasting, which saved me a significant amount of implementation time.


## Diebold-Mariano Test

### The Role of the Diebold-Mariano Test
In the context of comparing forecasting models, I initially thought the **Diebold-Mariano (DM)** test was redundant. Since MAPE quantifies the magnitude of forecast errors and Directional Accuracy assesses the correctness of the predicted direction, I assumed they gave a clear indication of which model performed better on the test set. However, I did not realize that these two metrics, by themselves, do not indicate whether an observed performance difference is statistically significant.

This is where the DM test becomes essential. The DM test determines if the difference in predictive accuracy between two competing forecasts is **statistically significant**. It directly addresses the question, "Is the observed superior performance of Model A over Model B genuine, or is it likely due to random chance?" The DM test analyzes the loss differentials (the difference in errors) between the two models to provide a p-value, which helps in making a statistically backed judgment.

For example, all my models had very similar MAPE values for a 1-week-ahead forecast, which could give the impression that they performed similarly. The DM test can help determine whether this slight difference is statistically significant enough to conclude that a given model is truly superior confidently.

In essence, MAPE and Directional Accuracy provide **point estimates** of performance, while the Diebold-Mariano test provides a **measure of confidence** in the difference between those performance estimates.

### Comparing Metrics (MAPE vs. MAE/RMSE)
I was concerned that my models were evaluated using MAPE and Directional Accuracy as primary performance metrics, whereas the DM test typically uses either MAE or RMSE to compute the loss. The question was whether the DM test still makes a fair and correct comparison.

It turned out that the DM test does not care how a model was trained. It only looks at the **final output**. The test defines predictive accuracy based on a chosen loss function (like MAE or RMSE). The test's only job is to answer: "Based on these two sets of forecast errors, is the average loss from Model A significantly different from the average loss of Model B?”

Using MAE or RMSE for the DM test is standard because they are robust, well-behaved statistical loss functions. The interpretation of the test results is simply that Model A's forecasts are significantly better than Model B's in terms of minimizing Mean Absolute Error (MAE) (or whichever loss function was used in the test).


## Deep Copy
When I ran the `train_and_forecast_with_stacked_ensemble_model()` function to obtain the final performance metrics and prediction output, an error occurred: `TypeError: cannot pickle 'Graph' object`. After some research, I learned that the error was due to the LSTM/GRU model being a Keras model.

Specifically, `copy.deepcopy()` often works by pickling (serializing) an object, then unpickling (deserializing) it to create a new copy. A compiled Keras model contains a complex computation graph ('Graph') that is tied to the live Python runtime and generally cannot be pickled. When `copy.deepcopy()` is called on the Keras wrapper, it tries to deep-copy all its attributes, including the trained Keras model, which causes the pickling to fail.
The solution was to implement a custom `__deepcopy__` method in the LSTM/GRU wrapper class. This method manually creates a new instance of the wrapper, rebuilds the model architecture, and then transfers the trained weights from the original model to the new instance.

### The XGBoost Bug
I thought I had addressed the problem, but another one popped up from an unexpected place. This time, an XGBoost model failed to predict the last data point in the test dataset, which produced a `NaN` that was fed to the `LinearRegression` meta-mode

I investigated the `XGBForecaster` class, assuming the problem lay in its internal evaluation logic rather than the stacking function. I looked for an off-by-one error in how it calculates the number of walk-forward steps. However, a standalone XGBoost model correctly evaluated all 486 data points. My focus then shifted to the `StackedEnsembleForecaster` class’s `evaluate()` method. The retrained models were correctly assigned the test dataset and the combined train/validation datasets from the original models. The root cause, once again, lay with `copy.deepcopy()`. The bug was very subtle.

### Bug Trace
1. The `stacked_model.fit()` in the `train_and_forecast_with_stacked_ensemble_model()` 
calls the `StackedEnsembleForecaster.fit()` method
2. It calls the `model.fit()` method on the original `XGBForecaster` model.
3. The model calls its `_prepare_data()` method, which creates `self.X_train`, `self.y_train`, `self.X_valid`, `self.y_valid`, `self.valid_ds_transformed` from the validation dataset.
4. The `XGBForecaster` model is now trained and full of attributes.
5. The `stacked_model.evaluate()` in the `train_and_forecast_with_stacked_ensemble_model()` 
calls the `StackedEnsembleForecaster.evaluate()` method.
6. It calls the `copy.deepcopy()` method.
7. The bug! Because the `XGBForecaster` class does not have a Keras model, it is perfectly pickleable. The `copy.deepcopy()` method works, creating a clone of the already-trained `XGBForecaster` model, including all of its attributes.
8. The `model_retrained.fit()` method is called.
9. Inside, `XGBForecaster.fit()` method, it checks if not `self.X_train` or not `self.X_valid` expression.
10. The check is false because the `self.X_train` was copied over, so the `_prepare_data()` is skipped.
11. The `XGBForecaster.evaluate()` method runs the loop, using the old, copied `self.valid_ds_transformed`. Thus, the evaluation produces 485 data points, not 486.
12. The solution was again to add a `__deepcopy__ `method to `XGBForecaster` class.