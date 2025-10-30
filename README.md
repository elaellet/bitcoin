# Bitcoin Price Forecasting: A Comparative Analysis of Statistical, ML, and DL Models
This repository details a comprehensive project to forecast Bitcoin (`BTC`) prices. It implements and rigorously compares a suite of models —statistical (`ARIMA`), deep learning (`LSTM`, `GRU`), and machine learning (`XGBoost`) —culminating in a final stacked ensemble.

The project contains the complete, end-to-end workflow, including: data preprocessing, feature engineering, model implementation, in-depth evaluation (using `MAPE`, `Directional Accuracy`, and `Diebold-Mariano` statistical tests), and final future price prediction.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Workflow](#project-workflow)
- [Data Sources](#data-sources)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Build](#build)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)


## Project Overview
This project explores time-series forecasting of Bitcoin's closing price at specific future horizons (e.g., 7-day and 30-day ahead). It implements and compares a comprehensive suite of models to determine their predictive effectiveness, including:

- **Naive Model**: A persistence model used as a baseline.
- **Statistical Model**: `ARIMA`
- **Deep Learning Models**: `LSTM` and `GRU`
- **Machine Learning Model**: `XGBoost`
- **Ensemble Model**: A stacked model combining predictions from the base models.

Model performance is rigorously evaluated on a hold-out test set using a `walk-forward validation` strategy. Key metrics include `Mean Absolute Percentage Error (MAPE)` and `Directional Accuracy (DA)`, with the `Diebold-Mariano (DM) test` used to assess the statistical significance of performance differences between the models.


## Project Workflow
The project follows a six-step workflow. High-level summaries are provided below, with links to detailed analyses, logs, and visualizations in the `/docs` directory.

1. **Preprocessing & Baseline**: Raw `BTC` `OHLCV` data is cleaned, feature-engineered (e.g., `log returns`, `technical indicators`, `macroeconomic features`), and preprocessed. This includes handling missing values, verifying data integrity, imputing outliers, and scaling features. A `naive` persistence model is established as a baseline for comparison.
   ➡️ **[Preprocessing](./docs/01_preprocessing.md)**

2. **Statistical Model (ARIMA)**: A classical statistical approach. The time series is first tested for stationarity using the `Augmented Dickey-Fuller (ADF)` test to determine the necessary order of differencing (`d`). The `ACF` and `PACF` plots are then inspected to estimate reasonable search ranges for the `p` (autoregressive) and `q` (moving average) terms. A grid search is performed within these ranges to find the optimal (`p`, `d`, `q`) parameters.
   ➡️ **[ARIMA](./docs/02_arima.md)**

3. **Recurrent Models (LSTM/GRU)**: Deep learning models adept at capturing temporal sequences. The time series data is transformed into supervised learning windows, where a sequence of `N` past time steps (the `input_window`) is used to predict the target at `M` steps in the future (the `target_window`). Both `Long Short-Term Memory (LSTM)` and `Gated Recurrent Unit (GRU)` models are implemented. Their architectures are tuned by searching for the optimal combination of hyperparameters—such as number of layers, hidden units, dropout rate, and learning rate—by minimizing the validation loss (`Huber`) and tracking validation metrics (`MAE`).
   ➡️ **[RNN](./docs/03_rnn.md)**

4. **Tree-Based Model (XGBoost)**: A machine learning model. `XGBoost (Extreme Gradient Boosting)` is adapted for time series by transforming the data into a tabular structure. Features (`X`) are created by lagging and windowing the entire set of past values (e.g., prices, technical indicators) to predict the future target price (`y`) at a specific horizon. An `XGBoost` regressor is trained for each forecast horizon (a direct multi-step strategy). Hyperparameters—such as `n_estimators`, `learning_rate`, `max_depth`, and `subsample`—are tuned via a grid search to optimize performance on the validation set.
   ➡️ **[XGBoost](./docs/04_xgb.md)**

5. **Stacked Ensemble Model**: A meta-model that combines the predictive strengths of the individual base forecasters. A stacked ensemble model (using `Linear Regression`) is trained on the out-of-sample validation predictions from the base models (e.g., `GRU` and `XGBoost`) to find their optimal predictive weighting.
   ➡️ **[Ensemble](./docs/05_ensemble.md)**

6. **Evaluation & Statistical Testing**: All models are rigorously evaluated on a hold-out test set using a `walk-forward validation` strategy. Performance is quantitatively compared using `Mean Absolute Percentage Error (MAPE)` and `Directional Accuracy (DA)`. Finally, the `Diebold-Mariano (DM)` test is used to assess the statistical significance of the forecast differences.
   ➡️ **[Evaluation](./docs/06_evaluation.md)**


## Data Sources

This project utilizes three primary sources of data:

* **Bitcoin (`BTC`) `OHLCV` Data (Kaggle):** 1-minute interval data (`Open`, `High`, `Low`, `Close`, `Volume`) for the `BTC/USD` pair from Bitstamp, spanning from January 2012 to the present.
    * **Source:** [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
    * **License:** `CC BY-SA 4.0`
* **Federal Funds Effective Rate (`FEDFUNDS`):** Monthly economic data obtained from [FRED, Federal Reserve Economic Data](https://fred.stlouisfed.org/series/FEDFUNDS).
* **M2 Money Supply (`M2SL`):** Weekly data on the M2 money stock, also obtained from [FRED, Federal Reserve Economic Data](https://fred.stlouisfed.org/series/M2SL).


## Getting Started
Follow these instructions to get the project up and running on your local machine.

### Prerequisites
- Python 3.12+
- Git
- `pip`
- A virtual environment tool like `venv`.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r 'requirements.txt'
   ```

## Build
To compile the project into a single executable file, run this command from the project's root directory:

```bash
pip install pyinstaller
pyinstaller --onefile --name "btc_analyzer" --add-data "data/raw:data/raw" --add-data "data/external:data/external" --add-data "config.yaml:." main.py
```


## Project Structure
```
├── data/
│   ├── cleaned/
│   ├── external/ 
│   ├── metadata/
│   └── raw/
│
│── docs/
│   ├── 01_preprocessing.md
│   ├── 02_arima.md
│   ├── 03_rnn.md
│   ├── 04_xgb.md
│   ├── 05_ensemble.md
│   └── 06_evaluation.md
│
├── notebooks/
│   ├── 00_pipeline.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_naive.ipynb
│   ├── 03_arima.ipynb
│   ├── 04_lstm.ipynb
│   ├── 05_gru.ipynb
│   ├── 06_xgb.ipynb
│   └── 07_evaluation.ipynb
│
├── outputs/
│   ├── plots/
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arima.py
│   │   ├── base.py
│   │   ├── ensemble.py
│   │   ├── gru.py
│   │   ├── lstm.py
│   │   ├── naive.py
│   │   └── xxgb.py
│   │   
│   ├── __init__.py
│   ├── analysis.py
│   ├── data_cleaner.py
│   ├── data_loader.py
│   ├── data_splitter.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── utils.py
│   └── visualization.py
│
├── .gitignore
├── config.yaml
├── README.md
├── requirements.txt
└── main.py
```


## Results
Here is the performance of the models on the test dataset:

### 📈 1-Week Ahead Forecast Results

| Model               | MAPE (%) | Directional Accuracy (%) | DM Test (vs. Baseline) |
|---------------------|----------|--------------------------|------------------------|
| Baseline            | 5.04     | 47.2                     | -                      |
| ARIMA               | 5.36     | 52.9                     | p-value >= 0.05        |
| LSTM                | 5.04     | 50.8                     | p-value >= 0.05        |
| GRU                 | 5.04     | 50.8                     | p-value >= 0.05        |
| XGB                 | 5.13     | 51.8                     | p-value >= 0.05        |
| Stacking (LSTM+XGB) | 5.20     | 52.1                     | p-value >= 0.05        |
| Stacking (GRU+XGB)  | 5.11     | 54.7                     | p-value >= 0.05        |

### 🗓️ 1-Month Ahead Forecast Results

| Model               | MAPE (%)  | Directional Accuracy (%) | DM Test (vs. Baseline) |
|---------------------|---------- |--------------------------|------------------------|
| Baseline            | 9.84      | 42.3                     | -                      |
| ARIMA               | 10.31     | 49.0                     | p-value >= 0.05        |
| LSTM                | 9.84      | 50.3                     | p-value <  0.05        |
| GRU                 | 9.83      | 50.3                     | p-value <  0.05        |
| XGB                 | 9.90      | 52.1                     | p-value >=  0.05       |
| Stacking (LSTM+XGB) | 9.83      | 54.2                     | p-value >= 0.05        |
| Stacking (GRU+XGB)  | 9.82      | 57.7                     | p-value >= 0.05        |

### 📊 Diebold-Mariano Test p-value Matrix (1-Week Ahead)
|                     | Baseline | ARIMA     | LSTM     | GRU      | XGB      | Stacking (LSTM+XGB) | Stacking (GRU+XGB) |
|---------------------|----------|-----------|----------|----------|----------|---------------------|--------------------|
| Baseline            | -        | 0.99      | 0.19     | 0.20     | 0.98     | 0.93                | 0.68               |
| ARIMA               | 0.99     | -         | **0.00** | **0.00** | **0.04** | **0.04**            | **0.01**           |
| LSTM                | 0.19     | **0.00**  | -        | 0.33     | 0.97     | 0.96                | 0.76               |
| GRU                 | 0.20     | **0.00**  | 0.33     | -        | 0.97     | 0.96                | 0.76               |
| XGB                 | 0.98     | **0.04**  | 0.97     | 0.97     | -        | 0.43                | 0.22               |
| Stacking (LSTM+XGB) | 0.93     | **0.04**  | 0.96     | 0.96     | 0.43     | -                   | 0.11               |
| Stacking (GRU+XGB)  | 0.68     | **0.01**  | 0.76     | 0.76     | 0.22     | 0.11                | -                  |

### 📊 Diebold-Mariano Test p-value Matrix (1-Month Ahead)
|                     | Baseline | ARIMA     | LSTM     | GRU      | XGB      | Stacking (LSTM+XGB) | Stacking (GRU+XGB) |
|---------------------|----------|-----------|----------|----------|----------|---------------------|--------------------|
| Baseline            | -        | 0.99      | **0.03** | **0.04** | 0.99     | 0.30                | 0.37               |
| ARIMA               | 0.99     | -         | **0.01** | **0.01** | **0.03** | **0.00**            | **0.01**           |
| LSTM                | **0.03** | **0.01**  | -        | **0.04** | 0.99     | 0.31                | 0.40               |
| GRU                 | **0.04** | **0.01**  | **0.04** | -        | 1.00     | 0.40                | 0.53               |
| XGB                 | 0.99     | **0.03**  | 0.99     | 1.00     | -        | 0.07                | **0.02**           |
| Stacking (LSTM+XGB) | 0.30     | **0.00**  | 0.31     | 0.40     | 0.07     | -                   | 0.67               |
| Stacking (GRU+XGB)  | 0.37     | **0.01**  | 0.40     | 0.53     | **0.02** | 0.67                | -                  |

### Key Findings
🏆 **Overall Model Performance**
1. **Best Model for Direction**: The `Stacking (GRU+XGB)` model is the clear winner for predictive direction. It achieves the highest `Directional Accuracy (DA)` in both the 1-week (54.7%) and 1-month (57.7%) forecasts. It also has the lowest `MAPE` (best error) for the 1-month forecast (9.82%).
2. **Worst Performing Model (`ARIMA`)**: The `ARIMA` model consistently performs the worst among the advanced models. It has the highest `MAPE` (worst error) in both the 1-week (5.36%) and 1-month (10.31%) forecasts. The `Diebold-Mariano (DM)` test matrices confirm that its forecasts are statistically significantly different from all other advanced models (`LSTM`, `GRU`, `XGB`, and `Stacking` models), with `p-values` all $\le 0.04$
3. **Baseline Performance**: The `Baseline` model is consistently the worst performer in terms of `Directional Accuracy`, scoring worse than a random guess (50%) at both 1-week (47.2%) and 1-month (42.3%) horizons.

📉 **1-Week Ahead Forecast**
- **No Significant Improvement**: This is the most critical finding for the 1-week horizon. According to the `DM test`, no model (including the top-performing `Stacking` models) provides a statistically significant improvement in forecast accuracy over the `Baseline` (all `p-values` >= 0.05).
- **Best vs. Worst Error**: The `Baseline` and `GRU` models are tied for the lowest `MAPE` (5.04%). As noted, the `ARIMA` model has the highest `MAPE` (5.36%). The best-performing `Stacking (GRU+XGB)` model has an `MAPE` of 5.11%.

📈 **1-Month Ahead Forecast**
- **Statistical Significance Found**: Unlike the 1-week forecast, two models show a statistically significant improvement over the `Baseline`:
  - `LSTM` (`p-value` < 0.05)
  - `GRU` (`p-value` < 0.05)

🤔 **A Key Contradiction**: A crucial finding is that while the `Stacking (GRU+XGB)` model has the best metrics (lowest `MAPE` and highest `DA`), its improvement over the `Baseline` is not statistically significant (`p-value` >= 0.05). This suggests that, while it performed better on average, its improvement wasn't sufficient to pass the statistical test. In contrast, the standalone `LSTM` and `GRU` models consistently outperformed the `Baseline`.

🧭 **Directional Accuracy (`DA`) vs. Error (`MAPE`)**
A significant finding is the difference in what the metrics say:

- **DA**: All advanced models (`ARIMA`, `LSTM`, `GRU`, `XGB`, `Stacking`) are clearly better than the `Baseline` at predicting the direction of the price movement.
- **MAPE**: With the apparent exception of the poorly performing `ARIMA` model, all other models (including the `Baseline`) have very similar error magnitudes (`MAPE`).

⚠️ **Model Limitation: Residual Analysis**
- **All Models Fail Diagnostic Checks**: A residual analysis was performed on all models for both forecast horizons. The `Ljung-Box Test` for all models returned a `p-value` < 0.05 (e.g., `p=0.0000`).

- **Implication**: This result rejects the null hypothesis (that residuals are random and independent). This means there is significant `autocorrelation` left in the errors.

- **Conclusion**: None of the models—from the `Naive` baseline to the `Stacking` ensembles—are fully capturing the information in the time series. This suggests all models could be improved by incorporating more relevant features.

💡 **Conclusion**: The primary value of the advanced models (especially the `Stacking` models) is not in reducing the magnitude of the forecast error (`MAPE`), but in correctly predicting the direction (`DA`) of the future price.


## Future Work
**Implement a Command-Line Interface (CLI)**: 
Currently, `main.py` runs the entire pipeline. A major improvement would be to add a CLI (using a library like `argparse` or `typer`) to `main.py`. This would allow developers to run specific parts of the project, similar to the functionality shown in the screenshot. For example:
- #### Example: Train only the LSTM model
```bash
python 'main.py' train --model 'lstm'
```
- #### Example: Run only the evaluation script
```bash
python 'main.py' evaluate
```


**Advanced Feature Engineering**:
- **On-Chain Data**: Integrate Bitcoin-native on-chain metrics (e.g., hash rate, transaction volume, active addresses), which are known to be leading indicators.
- **Sentiment Analysis**: Incorporate sentiment data by processing financial news headlines or social media (like `X/Twitter`) to capture market mood.

**Advanced Model Architectures**:
- **Transformer Models**: Implement and evaluate `Transformer`-based architectures, which have become the state-of-the-art for sequence data and may capture longer-term dependencies more effectively than `LSTMs/GRUs`.

**API Deployment**:
- Develop a simple `REST API` (using a framework like `FastAPI` or `Flask`) to serve predictions from the best-performing model (e.g., the `Stacking (GRU+XGB)` for 1-month direction), enabling real-world integration or a live-trading simulation.


## Acknowledgements
- The code structure for the`ARIMA` evaluation was adapted from Jason Brownlee's blog post, Time Series Forecast Case Study with Python: Monthly Armed Robberies in Boston.
- I would like to credit Aurélien Géron's book, 'Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow', which provided the conceptual foundation for the time series windowing techniques and the baseline architecture for the `RNN` models.
- A basic understanding of time series analysis concepts was gained from the online textbook, Forecasting: Principles and Practice (3rd Ed) by Rob J Hyndman and George Athanasopoulos.