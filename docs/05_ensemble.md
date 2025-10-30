## Ensemble
The LSTM/GRU performance on a monthly forecast using the validation dataset was unsatisfactory; its MAPE and Directional Accuracy did not outperform the naive baseline. This indicated I either had to find a new model or combine existing ones. This is where ensemble models shine, as they are a proven and robust technique for improving model performance by leveraging the complementary strengths of multiple models.

Even though the LSTM/GRU models did not beat the naive persistence model, their metrics were close to it. I decided to use these two RNN-based models as base models, since they excel at capturing temporal patterns and sequences. However, to create a strong ensemble, I needed another model that could provide new insights beyond what the RNNs captured. 

### XGBoost
Among the different types of models, the tree-based XGBoost caught my attention because it excels at uncovering complex, non-linear relationships across a wide range of features. Financial markets are complex; the relationship between past prices, volatility, and future prices is rarely linear.

XGBoost can effectively handle the hundreds of engineered features (lags, technical indicators, sentiment scores, etc.) and automatically determine which ones are most important. It provides clear feature importance scores, indicating which indicators (e.g., RSI, 50-day moving average) were most predictive. This is a massive advantage for interpretability compared to more black-box models like LSTMs. LSTM/GRU models already include a rich set of features — log returns, technical indicators, and macroeconomic indicators — generated through feature engineering.

### Bagging vs. Boosting vs. Stacking
There are three main types of ensembling, each with distinct characteristics:
- **Bagging**: Reduces **variance** and improves stability. It trains multiple models (usually of the same type) on different subsets of the data and averages their predictions. This helps smooth predictions and prevent overfitting.
- **Boosting**: Reduces **bias** and converts weak learners into strong ones. It builds models sequentially, with each new model focusing on correcting the errors made by its predecessors.
- **Stacking**: Improves overall **predictive accuracy** by combining the forecasts of different types of models. It uses a meta-model to learn the best way to combine the predictions from the base models, leveraging their complementary strengths.

Unlike bagging and boosting, which typically use homogeneous models, stacking is designed to use **heterogeneous** base models. Since my goal was to combine the temporal-pattern-finding strength of RNNs with the feature-driven, non-linear strength of XGBoost, the choice was clear.


## Stacking
Stacking is an ensemble method that improves overall predictive accuracy by combining the forecasts from different types of models. It operates on two levels:
- **Level 0 (Base Models)**: Multiple diverse models (e.g., an RNN, an XGBoost) that are trained independently.
- **Level 1 (Meta-Model)**: A single model (often a simple one, like `LinearRegression`) that learns the best way to combine the predictions from the base models.

The core idea is to leverage the complementary strengths of each base model, with the meta-model learning to trust each base model's predictions under different circumstances.

### The Stacking Fitting and Evaluation Process
The most critical part of stacking is how it uses the validation set to train the meta-model without **data leakage**. Training the meta-model on the same predictions from the data the base models were trained on would lead to severe overfitting.

The process is typically as follows:

1. **Tune Base Models (Level 0)**: Each base model (RNN, XGBoost) is optimized independently. This involves its own hyperparameter tuning, feature selection, and validation (e.g., using a standard train/validation split) to find the best-performing version of that model.

2. **Generate Meta-Model Training Data**: The meta-model must be trained on predictions that the base models have not seen during their own training. For time series, this is achieved with walk-forward validation on the training set.
    - Fold 1: Train base models on Weeks 1-52. Make predictions on Week 53.
    - Fold 2: Train base models on Weeks 2-53. Make predictions on Week 54.
    - ...and so on.

The collection of these out-of-sample predictions (for Week 53, Week 54, etc.) forms the feature set (X) for the meta-model. The actual values for those weeks form the target (y).

3. **Train the Meta-Model (Level 1)**: With the out-of-sample dataset from Step 2, the meta-model is now trained. It learns how to optimally combine the unbiased outputs of the base models. For example, it might learn that the RNN is more reliable for long-term trends while XGBoost is better at capturing short-term volatility.

4. **Final Evaluation on the Test Set**: To get the final, unbiased performance, the entire process is finalized:
    1. The final, optimized base models (from Step 1) are retrained on the entire original training set.
    2. These retrained base models generate their predictions on the new, unseen test set.
    3. These test predictions are fed as input to the trained meta-model (from Step 3).
    4. The meta-model generates the final ensemble prediction. This output is compared against the true test set values to get the final performance metrics.