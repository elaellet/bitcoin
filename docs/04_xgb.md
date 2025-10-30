## XGBoost

### Tree-Based Model for Time Series
A tree-based model inherently lacks a concept of temporal order or sequence. To use it for time-series data, the problem must be transformed from a sequence prediction task to a **tabular regression task**. This means the model doesn't look at the raw time series directly; instead, it uses a set of features derived from the series' past to predict a future point.

For example, a price forecast at time $t$ requires features from previous time steps, $t-1$, $t-2$, etc. These lag features are themselves a form of feature engineering and are typically used alongside other engineered features, such as technical indicators (e.g., RSI, SMA).

### XGBoost vs. LightGBM
When I decided to implement a stacked ensemble, I had to choose between **LightGBM** and **XGBoost** for the tree-based component, as both offer very high predictive accuracy, often comparable.
- LightGBM's most significant advantage is its speed; it's typically much faster to train and more memory-efficient than XGBoost. However, its default leaf-wise growth strategy carries a higher risk of overfitting, especially on smaller datasets.
- XGBoost is naturally more robust to overfitting due to its default level-wise growth and strong built-in regularization.

The growth strategies are the key difference. LightGBM's **leaf-wise strategy** grows the tree by splitting the single leaf that yields the maximum reduction in loss. This often leads to deeper, unbalanced trees that can achieve high accuracy but are prone to overfitting. XGBoost's **level-wise strategy** grows the tree by splitting all leaves at a given depth before moving to the next level. This results in a more balanced, stable, and robust tree, though it might occasionally be marginally less accurate than a perfectly tuned leaf-wise tree.

Given Bitcoin's notorious volatility and vulnerability to unpredictable events, overfitting to past noise can be fatal. XGBoost's robustness helps manage this risk. Since well-tuned LightGBM and XGBoost models often have very similar predictive accuracy, and my resampled datasets were not large enough for training speed to be the primary concern, XGBoost was the more prudent choice.