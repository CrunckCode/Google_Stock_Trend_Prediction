# Google Stock Trend Prediction (LSTM)

A recurrent neural network (stacked LSTM) in Keras that predicts Google's daily opening price from the previous 60 trading days, and plots predicted against real prices for January 2017.

## Data
- `Google_Stock_Price_Train.csv`: daily Open, High, Low, Close and Volume from 2012 to 2016 (about 1,258 trading days).
- `Google_Stock_Price_Test.csv`: 20 trading days of January 2017.

## Method
1. Use the Open price only and scale it to [0, 1] with `MinMaxScaler`.
2. Build supervised samples: each input is a window of the previous 60 days and the target is the next day's price (1,198 training samples).
3. Model: four LSTM layers of 50 units, each followed by 20% dropout, then a single Dense output. Adam optimiser, mean squared error loss, 100 epochs, batch size 32.
4. For testing, take the last 60 training days plus the test period as context, predict each of the 20 test days, and invert the scaling.
5. Plot the real and predicted price: `Google Stock Price Real VS Predicted.png`.

## Files
- `rnn.py` and `rnn.ipynb`: the same code as a script and as a notebook.
- `Google Stock Price Real VS Predicted.png`: the resulting chart.

## How to run
```
pip install numpy pandas matplotlib scikit-learn tensorflow
python rnn.py
```

## Notes and limitations
- The model follows the general direction of the price (the trend) rather than matching each day exactly. No error metric is recorded in the repository; the chart is the result.
- Predicting the next day from the previous 60 days of the same series is a pattern-learning exercise, not a trading strategy. It uses no other information (volume, news, market moves), and the 20-day test window is very short.
- This was a learning project, built to practise sequence modelling and time-window preparation.
