
#Import libraries
import pandas as pd
from pandas_datareader import data as pdr
import pandas_datareader as webreader
import yfinance as yf
import numpy as np
import datetime
import pandas_ta as ta
import talib as tb
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import tensorflow as tf
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import plot_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
yf.pdr_override()


def get_stock_prices(symbol: str, end : str):
    '''Fetches daily stock prices from Yahoo Finance for a specified stock symbol up to a given end date. 
    It then adds the stock symbol to the data, resets the index, prints a preview of the data, 
    and exports it to a CSV file. 
    The CSV file is saved with a filename based on the stock symbol and end date.
    
    Parameters:
    ------------  
    symbol (String): A string representing the stock symbol for which to fetch the prices.
    end (String): A string representing the end date until which to fetch the prices in the format 'YYYY-MM-DD'.

    Returns:
    --------.
    df.to_csv (dataframe): dataframe is saved as csv using comma as separator and index false
    
    '''
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d')
    df = pdr.get_data_yahoo(symbol, end=end_date)
    df['Symbol'] = symbol
    df.reset_index(inplace=True)
    end = end.replace('-', '')
    print(df.tail())
    return df.to_csv(f'./data/{symbol}_{end}.csv', sep=',', index=False)


def technical_indicators(df_path: str):
    '''The function performs various technical indicator 
    calculations on the stock data read from the CSV file.
    Parameters:
    ------------
    df_path (String): A string representing the file path of the CSV file containing the stock data.

    Returns:
    --------
    df (dataframe): Updated DataFrame with technical indicators.
    '''    
    df = pd.read_csv(df_path, sep=',', index_col='Date')
    df.ta.log_return(cumulative=True, append=True)
    df.ta.percent_return(cumulative=True, append=True)
    df['EMA_50'] = df.ta.ema(length=50, append=True)
    df['EMA_100'] = df.ta.ema(length=100, append=True)
    df['SMA_50'] = df.ta.sma(length=50, append=True)
    df['SMA_100'] = df.ta.sma(length=100, append=True)
    df['MACD'] = tb.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)[0]
    df['RSI'] = tb.RSI(df['Close'].values, timeperiod=14)
    df.rename(columns = {'CUMLOGRET_1':'Log Return', 'CUMPCTRET_1':'% Return'}, inplace = True)
    print(df.tail())
    return df


def plot_split_data(df, n:int):
    '''The function generates and displays plots 
    related to Moving Average Convergence Divergence (MACD) and 
    Relative Strength Index (RSI) indicators. 
    It then returns a candlestick chart plot 
    with the specified customizations.
    
    Parameters:
    ------------
    df (DataFrame): A pandas DataFrame containing stock data.
    n (int): An integer representing the number of periods of data to consider.


    '''
    dir = "./img/"
    company = df['Symbol'][0]
    df = df.tail(n)
    df.index = pd.to_datetime(df.index)  # Convert index to DatetimeIndex
    dates = df.index.strftime('%B, %Y')  # Format dates
    date_1 = dates[0]
    date_2 = dates[n-2]
    
    macd_plot = mpf.make_addplot(df["MACD"], panel=3, color='fuchsia', title="MACD")
    macd_hist_plot = mpf.make_addplot(df["MACD"], type='bar', panel=3) 
    df['signal'] = df["MACD"].ewm(span=9).mean()
    macd_signal_plot = mpf.make_addplot(df["signal"], panel=3, color='b')
    rsi_plot = mpf.make_addplot(df['RSI'], panel=2, color='red', title='RSI')
    plots = [macd_plot, macd_signal_plot, macd_hist_plot, rsi_plot]

    plt.show()

    return mpf.plot(df, type='candle', style='yahoo', volume=True, mav=(50, 25),
                    tight_layout=True, figratio=(10, 5), addplot=plots,
                    title=f"{company} {date_1} - {date_2}",
                    panel_ratios=(.35, .1, .15, .15),
                    datetime_format='%b %d',
                    xrotation=0,
                    returnfig=True)




def detect_outliers(df, column: str, min_quantile: float, max_quantile: float):
    '''
    The function detects outliers in the specified column of the input DataFrame 
    based on the given quantiles and returns a DataFrame containing those outliers.
    
    Parameters:
    ------------
    df (DataFrame): A pandas DataFrame containing the data.
    column (string): A string representing the column name in the DataFrame where outliers will be detected.
    min_quantile (float): Represents the lower quantile value used to calculate the lower bound for outlier detection. 
                            It determines the threshold below which data points are considered outliers.
    max_quantile (float): Represents the upper quantile value used to calculate the upper bound for outlier detection. 
                        It determines the threshold above which data points are considered outliers.

    Returns:
    ---------
    outliers (DataFrame): A new DataFrame containing the outliers.
    '''
    q1 = df[column].quantile(min_quantile)
    q3 = df[column].quantile(max_quantile)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers



def handle_outliers(df, column):
    '''
    The function returns the filtered DataFrame.
    
    Parameters:
    -----------
    df (DataFrame): The input DataFrame containing the data.
    column (str): The column name in the DataFrame for which outliers will be handled.

    Returns:
    ---------
    df(DataFrame): The filtered DataFrame with the rows containing outliers removed.

    '''
    outliers = detect_outliers(df=df, 
                            column='Close',
                            min_quantile=0.25, 
                            max_quantile=0.75)
    return df[df[column] <= outliers[column].min()]

def drop_columns(df):
    columns = ['Symbol', 'Volume', 'Open', 'High', 'Low']
    df = df.drop(columns=columns, axis=1)
    return df


def data_preprocess(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    scaled_data = pd.DataFrame(scaled_data, columns=df.columns, index=df.index).dropna()
    return scaled_data


def lstm_model_train(df: pd.DataFrame,
                  symbol: str = 'GOOG', 
                  selected_activation: str = 'relu', 
                  selected_learning_rate: float = 0.001, 
                  selected_epochs: int = 100, 
                  selected_batch_size: int = 30, 
                  selected_loss_function: str = 'mean_squared_error', 
                  plot_lstm_model: bool = False):
  ''' 
  Implements a Long Short-Term Memory (LSTM) model for stock price prediction based on given features. 
  The function takes in a pandas DataFrame containing historical stock prices and some technical indicators alongside 
  optional hyperparameters for the model by default, such as the selected optimizer, activation function, learning rate, 
  number of epochs, batch size, and loss function.
  The model is used to predict the test data with MSE and R^2 metrics computed. 
  The function returns a pandas DataFrame containing these metrics.
  If plot_lstm_model is set to True, the function also generates a visualization of 
  LSTM model and saves it to a png file.

  Parameters:
  ------------
  df (pd.DataFrame): A pandas dataframe containing the financial data to be used for analysis.
  features (list): List of features that is going to be considered in LSTM model.
  selected_optimizer (str, optional): Indicates the optimization algorithm to be used in training the LSTM model. Default is 'Adam'.
  selected_activation (str, optional): A string indicating the activation function to be used in the output layer of the LSTM model. Default is 'relu'.
  selected_learning_rate (float, optional): A float indicating the learning rate to be used in training the LSTM model. Default is 0.01.
  selected_epochs (int, optional): An integer indicating the number of epochs to be used in training the LSTM model. Default is 50.
  selected_batch_size (int, optional): An integer indicating the batch size to be used in training the LSTM model. Default is 32.
  selected_loss_function (str, optional): A string indicating the loss function to be used in training the LSTM model. Default is 'mean_squared_error'.
  plot_lstm_model (bool, optional): A boolean indicating whether or not to plot the LSTM model. Default is False

  
  Returns:
  ----------

  metrics_df (pd.DataFrame): A pandas dataframe containing the evaluation metrics of the LSTM model, 
  including MSE and R^2 metrics.

  '''

  stock_name = symbol
  X = np.array(df.drop('Close', axis=1))
  Y = np.array(df['Close'].values)

            
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

  timesteps = 10
  n_features = X.shape[1]
  
  X_train = np.array([X_train[i:i+timesteps] for i in range(len(X_train) - timesteps + 1)])
  X_test = np.array([X_test[i:i+timesteps] for i in range(len(X_test) - timesteps + 1)])

  Y_train = Y_train[timesteps - 1:]
  Y_test = Y_test[timesteps - 1:]

  
  X_train = X_train.reshape(X_train.shape[0], timesteps, n_features)
  X_test = X_test.reshape(X_test.shape[0], timesteps, n_features)

  selected_activation ='relu'
  selected_learning_rate = 0.01 
  selected_loss_function ='mse'
# Define the LSTM model
  model = Sequential()
  model.add(LSTM(units=200, 
                 return_sequences=True, 
                 input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(0.2))
  model.add(Dense(1, 
                  activation=selected_activation))

  optimizer = Adam(learning_rate=selected_learning_rate)
  model.compile(
      loss=selected_loss_function, 
      optimizer=optimizer,
      metrics = ['mse', 'mae']
            )

  # Model training
  history = model.fit(X_train, 
                      Y_train, 
                      epochs=selected_epochs, 
                      batch_size=selected_batch_size, 
                      verbose=1)
  with open('../models/lstm_model.pkl', 'wb') as file:
    pickle.dump(history.model, file)              

  return history, X_test, Y_test



  #reshape X_testing data
def reshape_test_data(df: pd.DataFrame) -> np.ndarray:
  '''
    Reshapes the test data for LSTM model prediction.

    Parameters:
    -----------
    df (pd.DataFrame): A pandas DataFrame containing the test data.

    Returns:
    --------
    np.ndarray: The reshaped test data in the form of a 3D numpy array.

  '''
  X = np.array(df.drop('Close', axis=1))
  timesteps = 10
  n_features = X.shape[1]
  X_test = np.array([X[i:i+timesteps] for i in range(len(X) - timesteps + 1)])
  X_test = X_test.reshape(X_test.shape[0], timesteps, n_features)
  return X_test


def predictions_as_array(df: pd.DataFrame, X_array: np.ndarray, model) -> np.ndarray:
    '''
    Converts predictions to an array and applies inverse scaling.

    Parameters:
    -----------
    df (pd.DataFrame): A pandas DataFrame containing the original data for scaling.
    X_array (np.ndarray): A numpy array representing the input data for prediction.
    model: The trained LSTM model used for prediction.

    Returns:
    --------
    np.ndarray: The predictions as a 1D numpy array.

    '''
    min_value = np.min(df['Close'])
    max_value = np.max(df['Close'])

    Y_pred = model.predict(X_array)
    Y_pred = Y_pred * (max_value - min_value) + min_value
    reshaped_Y_pred = Y_pred.reshape((len(X_array), 10))
    return reshaped_Y_pred[0]

def show_data_result(df: pd.DataFrame, y_predicted: np.ndarray, n: int, make_plot: bool = True) -> pd.DataFrame:
    ''' 
    Displays the data result with predicted values and optionally plots the data.

    Parameters:
    -----------
    df (pd.DataFrame): A pandas DataFrame containing the original data.
    y_predicted (np.ndarray): A numpy array representing the predicted values.
    n (int): The number of data points to display.
    make_plot (bool, optional): A boolean indicating whether to make a plot. Default is True.

    Returns:
    --------
    pd.DataFrame: A pandas DataFrame containing the data result.

    '''
    start_date = df.index[-1]
    dates = pd.date_range(start=pd.to_datetime(start_date) + pd.DateOffset(days=1), 
                            periods=len(y_predicted), 
                            freq='D')
    new_data = pd.DataFrame({'Close': y_predicted}, index=dates)
    data_result = pd.concat([df, new_data])
    data_result.index = pd.to_datetime(data_result.index)
    data_result.index = data_result.index.tz_localize(None)
    data_result = data_result.tail(n)
    if make_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data_result.index, data_result['Close'], color='black', label='Real today')
        ax.plot(data_result.index[-10:], data_result['Close'][-10:], color='red', label='Next 10 days')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Close Prices')
        ax.legend()

        # Display the plot
        plt.show()
    return data_result[['Close']]

if __name__ == '__main__':
    get_stock_prices('MSFT', '2023-06-01')




