'''
 build, train, and hyperparameter tune the LSTM model using MLflow and Keras.
 Basically:
 Import necessary libraries and utility functions from other modules.
 Define a function to create an LSTM model with hyperparameters.
 Define a function to load a trained model from a file.
 Define a function for running hyperparameter tuning using random trials.
 Define a function for hyperparameter tuning using grid search.


 Inside the main section, it sets up MLflow tracking, runs hyperparameter tuning using random trials 
 (run_trials_lstm_model) for 20 trials, and logs the results and the trained models to MLflow.
 Additionally, there's another function for hyperparameter tuning using grid search 
 (grid_search_lstm_model). However, it is not used in the main section.
 
'''


from model_utilities import technical_indicators
from model_utilities import handle_outliers, data_preprocess, drop_columns
from model_utilities import reshape_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from mlflow.models.signature import infer_signature

import pickle
import mlflow
import numpy as np
import pandas as pd
import mlflow.sklearn



def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def create_model(hidden_units, learning_rate, batch_size, epochs):
    model = Sequential()
    model.add(LSTM(units=hidden_units, 
                 return_sequences=True, 
                 input_shape=(1, 9)))
    model.add(Dropout(0.2))
    model.add(Dense(1, 
                  activation='relu'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
      loss='mse', 
      optimizer=optimizer,
      metrics = ['mse', 'mae']
            )
    return model

def run_trials_lstm_model(df_path, n_trials=20):
    # Perform n trials runs of the experiment
    for i in range(n_trials):
        hidden_units = np.random.choice([128, 256, 500])
        learning_rate =  np.random.choice([0.001, 0.01, 0.05])
        batch_size =  np.random.choice([10, 40, 60])
        epochs = np.random.choice([5, 10, 20, 30])

        data_raw = technical_indicators(df_path=df_path)
        data = handle_outliers(data_raw, 'Close')
        data = drop_columns(data)
        scaled_data = data_preprocess(data)

        X = np.array(scaled_data.drop('Close', axis=1))
        Y = np.array(scaled_data['Close'].values)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
        timesteps = 1
        n_features = X.shape[1]

        X_train_reshaped = reshape_data(X_train, timesteps=1)
        X_test_reshaped = reshape_data(X_test, timesteps=1)

        Y_train_reshaped = Y_train[timesteps - 1:]
        Y_test_reshaped  = Y_test[timesteps - 1:]

        model = create_model(hidden_units, learning_rate, batch_size,epochs)
        model.fit(X_train_reshaped, Y_train_reshaped)

        min_value = np.min(data_raw['Close'])
        max_value = np.max(data_raw['Close'])
        Y_pred = model.predict(X_test_reshaped)
        Y_pred * (max_value - min_value) + min_value
        reshaped_Y_pred = Y_pred.reshape((len(X_test_reshaped), 1))
        print(reshaped_Y_pred)
        print(Y_test_reshaped)
        rmse = mean_squared_error(Y_test_reshaped, reshaped_Y_pred, squared=False)
        r2 = r2_score(Y_test_reshaped, reshaped_Y_pred)
        signature = infer_signature(X_test, Y_pred)

        with mlflow.start_run():
            mlflow.set_tag("developer", "Javier")
            mlflow.log_param("train-data-path", "./data/AAPL_20230601.csv")
            mlflow.set_tag("model", "lstm")
            mlflow.log_param("hidden_units", hidden_units)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, 
                                    artifact_path="lstm_model", 
                                    signature=signature)
            mlflow.log_artifact(local_path="./models/lstm_model.pkl", artifact_path="models_pickle")

def grid_search_lstm_model(df_path):
    data_raw = technical_indicators(df_path=df_path)
    data = handle_outliers(data_raw, 'Close')
    data = drop_columns(data)
    scaled_data = data_preprocess(data)

    X = np.array(scaled_data.drop('Close', axis=1))
    Y = np.array(scaled_data['Close'].values)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    timesteps = 1
    n_features = X.shape[1]

    X_train_reshaped = reshape_data(X_train, timesteps=1)
    X_test_reshaped = reshape_data(X_test, timesteps=1)

    Y_train_reshaped = Y_train[timesteps - 1:]
    Y_test_reshaped  = Y_test[timesteps - 1:]

    param_grid = {
        'hidden_units': [200, 500],
        'learning_rate': [0.001, 0.01, 0.05],
        'batch_size':  [10, 40, 50],
        'epochs':  [10, 20, 40]
    }


    keras_estimator = KerasRegressor(build_fn=create_model)
    grid_search = GridSearchCV(keras_estimator, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_reshaped, Y_train_reshaped)

    with mlflow.start_run():
            mlflow.set_tag("developer", "Javier")
            mlflow.set_tag("model", "lstm")
            mlflow.log_param("train-data-path", "./data/AAPL_20230601.csv")
            signature = infer_signature(Y_test_reshaped, reshaped_Y_pred)
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(param, value)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", -grid_search.best_score_)
            mlflow.log_metric("r2", r2)
            mlflow.log_param('Best Params', grid_search.best_params_)
            mlflow.log_artifact(local_path="./models/lstm_model.pkl", artifact_path="models_pickle")