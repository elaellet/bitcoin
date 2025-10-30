## Standardization vs. Min-Max Scaling

### Scaling Log Returns for Neural Networks
After making the data stationary by computing log returns, I faced a critical decision on how to scale the data for the neural network. This scaling is essential for models like LSTMs and GRUs to train effectively. The two primary options were Min-Max scaling and Standardization.

**Min-Max scaling**, which scales data to a fixed range (e.g., `[0, 1]`) based on the absolute minimum and maximum values, is generally problematic for volatile financial returns. Bitcoin's log returns are characterized by extreme outliers—massive spikes and crashes. If a single crash (the minimum) and a single spike (the maximum) define the entire scaling range, all other data points will be compressed into a very narrow band. This squashing effect makes it difficult for the model to discern meaningful patterns, as the variance in the majority of the data becomes vanishingly small.

**Standardization**, on the other hand, scales the data so that the mean is 0 and the standard deviation is 1. Because this method relies on the mean and standard deviation, it is far more robust to outliers. While an extreme value will still influence these statistics, it does not absolutely dictate the entire scaling range. This ensures that the bulk of the data is distributed around 0, preserving its relative variance.

Given the extreme volatility and significant outliers in Bitcoin's log returns, Standardization was the clear choice. It provides more stable, reliable scaling, which is crucial for effective model training.


## Clipnorm vs. Clipvalue
Gradient clipping is a technique used to prevent the problem of exploding gradients. It works by constraining gradients during backpropagation to ensure they do not exceed a threshold. Two common methods for this are `clipvalue` and `clipnorm`.
- **`clipvalue` (Clipping by Value)**: This method caps each component of the gradient vector independently. For example, if the threshold is 0.5 and the gradient is `[1.2, 0.3]`, it becomes `[0.5, 0.3]`. This fundamentally alters the original direction of the gradient descent step. Since the gradient points the optimizer toward the minimum in the loss landscape, distorting this direction can destabilize the training process.
- **`clipnorm` (Clipping by Norm)**: In contrast, this method examines the overall L2 magnitude (length) of the entire gradient tensor. If this magnitude exceeds the threshold, `clipnorm` scales the entire vector down proportionally so its L2 norm matches the threshold. This process preserves the original direction of the gradient; it effectively just shortens the step size of the weight update. This ensures the model continues to learn along the correct path (steepest descent) with a smaller step size, leading to more stable training.

Therefore, I decided to use `clipnorm` to prevent exploding gradients, as it does not corrupt the gradient's directional information.


## L1 Regularization vs. L2 Regularization

### Regularization vs. Gradient Clipping
While selecting regularization hyperparameters, I needed to clarify the relationship between L1/L2 regularization and `clipnorm`. I initially thought they were related, but they are independent techniques that solve different problems.
- **Gradient Clipping (`clipnorm`)** is a stabilization technique. It prevents the exploding gradients problem during training by rescaling gradients if their magnitude exceeds a threshold, ensuring a stable learning process.
- **L1/L2 Regularization** is an overfitting prevention technique. Overfitting occurs when a model learns the training data too well, including its noise. Regularization simplifies the model by penalizing large weights, forcing it to focus on the most essential patterns and improving its generalization to new, unseen data.

### L1 (Lasso) vs. L2 (Ridge)
The key difference between these regularization methods lies in how they penalize the model's weights.
- **L1 regularization (Lasso)**: This method adds a penalty proportional to the absolute value of the weights. This characteristic can shrink some weights to precisely zero, effectively performing feature selection, by removing the influence of less critical features. 
- **L2 regularization (Ridge)**: This method adds a penalty proportional to the squared value of the weights. This encourages all weights to be small and evenly distributed, pushing them closer to zero but rarely making them exactly zero. This results in a diffuse model in which many features contribute only a small amount to the final prediction, rather than a sparse model (like L1) that relies on only a few features.

### Choosing L2 for Financial Time Series
For LSTM/GRU models analyzing financial time series, L2 is highly desirable. Price movements are driven by a multitude of complex, interacting factors. L2's approach, which creates a diffuse model with small weights, often leads to better generalization and more robust performance on noisy data. L1, by contrast, might discard useful, albeit subtle, information by forcing weights to zero.
	
The complex and noisy nature of Bitcoin price data means that a model relying on a wide array of small inputs (as encouraged by L2) is likely more robust than one that selects only a few, potentially spurious, strong signals (as encouraged by L1). This makes L2 a smoother, more stable form of regularization, particularly well-suited to recurrent connections in LSTMs and GRUs.

## Batch Normalization vs. Layer Normalization
Batch Normalization (BN) and Layer Normalization (LN) are both techniques that stabilize and accelerate deep learning training by normalizing layer activations. Their fundamental difference lies in which activations they normalize together.

### Normalization Axis
- **Batch Normalization (BN)**: Normalizes across the batch. For a given feature (e.g., a neuron), it computes the mean and variance of its activations across all samples in the current mini-batch. This makes BN highly dependent on the batch size; small batches produce unstable statistics, harming performance.
- **Layer Normalization (LN)**: Normalizes all features (layer). For a single sample, it calculates the mean and variance of all feature activations within that layer. This makes LN independent of the batch size, as all calculations are contained within a single sample.

### LN for RNNs
RNNs process sequences of variable lengths. If BN were applied, the statistics (mean/variance) would be calculated across the batch at each time step. The statistics for the 1st time step in a sequence would differ from those for the 20th time step. This would require the model to learn separate scaling parameters for every single time step, which is inefficient and fails to generalize to sequences of lengths not seen during training. LN solves this by normalizing within a single sample and time step (ie, across the hidden state's neurons). The normalization is consistent regardless of the sequence length or the statistics of other samples in the batch, making it stable and effective for sequential data.


## Custom MAPE Metrics and Huber Loss
While running hyperparameter tuning for LSTM/GRU models with the Keras Tuner, I needed to decide whether a custom-based MAPE metric would provide a meaningful improvement. The models ingest raw Bitcoin prices, convert them into log returns, and use these log returns for training. Therefore, simply adding a built-in `MAPE` metric to the callbacks (like `EarlyStopping` or `ReduceLROnPlateau`) would not work. It would be calculating MAPE on log returns (a percentage error of a percentage change), which is neither intuitive nor useful.

The distinction between losses and metrics was critical here.
- **Losses** are used by gradient descent to train a model. They must be **differentiable** and have **non-zero gradients** everywhere. 
- **Metrics** are used to evaluate and monitor a model. They can be **nondifferentiable**, have **zero gradients** everywhere, and are not used for calculating gradient updates. 

I initially considered using MAE (Mean Absolute Error) as the loss function. Since the model trains on log returns, minimizing the MAE of log returns (`abs(log_true - log_pred)`) serves as a proxy for minimizing the percentage error, since the log difference is a close approximation of a percentage change.

However, Bitcoin's returns exhibit fat tails, meaning extreme events occur far more often than a normal distribution would suggest. This presents a problem for standard loss functions:
  1. **MSE (Mean Squared Error)** squares the error. An outlier (a large error) is squared, resulting in a massive loss. This single, enormous loss value can dominate the training, causing unstable gradient updates as the model becomes obsessed with predicting rare events at the expense of normal-day accuracy.
  2. **MAE (Mean Absolute Error)** is robust to outliers because the loss scales linearly. A large error is just a large loss, not an exponentially large one. However, the MAE function has a sharp corner at zero (its gradient is undefined), which can sometimes make optimization less stable as it approaches the minimum.
  3. **Huber Loss** combines the best of both. For small errors (when the prediction is close to the actual), it behaves like MSE, providing a smooth, stable gradient for fine-tuning. For large errors (the outliers), it behaves like MAE, becoming linear and robust. This prevents the fat-tailed events from hijacking the training process.

Therefore, the optimizer uses the Huber loss on log returns to calculate gradients and update the model’s weights, so the model’s entire learning process is driven by minimizing this value. In contrast, metrics are just for reporting; their values are never used to calculate gradients. At the end of each epoch, the model pauses, uses the current weights to make predictions on the training and validation sets, and then calculates the metric values, printing them to the screen.

This means that simply monitoring a custom MAPE metric on prices will not directly change the model's final weights. However, it can indirectly improve the model when used with callbacks.

The model's training is driven by minimizing Huber loss on log returns, but the business goal is to minimize MAPE on prices. These two objectives are highly correlated, but not perfectly. A situation could arise where the validation loss (Huber) on log returns continues to improve slightly, while the validation MAPE on prices has already started to worsen (a sign of overfitting to the log-return-based loss)

By using a custom MAPE metric on prices with the `EarlyStopping` callback, it's possible to stop the model at the point of peak practical performance (lowest price-based MAPE), even if the proxy loss function (Huber on log returns) could still be improved. This indirectly results in a better, more useful final model.