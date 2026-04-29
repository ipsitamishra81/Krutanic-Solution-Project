Stock Price Forecasting using Transformer (PyTorch)

Overview

This project implements a Transformer-based deep learning model to predict stock prices using historical time series data.

The model learns temporal patterns from past prices and forecasts future values. Unlike traditional LSTM approaches, this project uses a Transformer architecture to better capture long-range dependencies.

Key Features

- Transformer-based time series forecasting (PyTorch)
- Sequence modeling using sliding window approach
- Data normalization using MinMaxScaler
- Model evaluation using RMSE
- Visualization of Actual vs Predicted prices


Dataset

- Source: Yahoo Finance ("yfinance")
- Stock: Apple Inc. (AAPL)
- Time Period: 2010 – 2024
- Feature used: Closing Price

Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- yfinance

Model Architecture

- Input embedding layer
- Transformer Encoder (multi-head attention)
- Fully connected output layer

Workflow

1. Download stock data using yfinance
2. Normalize data using MinMaxScaler
3. Convert time series into sequences
4. Train Transformer model
5. Evaluate predictions using RMSE
6. Visualize results

Results

- The model successfully captures overall trend and direction
- Predictions are smoother than actual prices (expected behavior)
- Sharp spikes are not perfectly predicted due to market volatility

Evaluation Metric

Root Mean Squared Error (RMSE) is used:

- Measures difference between predicted and actual prices
- Lower RMSE indicates better performance

Limitations

- Stock prices are inherently noisy and hard to predict
- Model smooths out sudden spikes
- Sentiment and external factors are not included in this version

Future Improvements

- Add news sentiment analysis
- Multi-feature input (Open, High, Low, Volume)
- Multi-step future prediction
- Deploy using Streamlit dashboard

How to Run

1. Install dependencies

pip install numpy torch matplotlib scikit-learn yfinance

2. Run training

python your_script_name.py

Key Learnings

- Transformer models can be applied to time series forecasting
- Proper preprocessing (scaling + sequences) is crucial
- Model tuning significantly impacts performance
- Financial data is noisy and requires careful interpretation

Author

Ipsita Mishra
