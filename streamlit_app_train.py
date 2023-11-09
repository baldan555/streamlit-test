import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import plotly.express as px
import mysql.connector

import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU

from itertools import cycle

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# Define a Streamlit app
st.title("Oil Price Prediction")

# Connect to the MySQL database
connection = mysql.connector.connect(
    host="217.21.73.32",
    user="u1801671_dmbsupredict",
    password="r.Sv[q!{=gl6",
    database="u1801671_dmbsupredict"
)

# Check if the database connection is successful
if connection.is_connected():
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM harga_argus")
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    cursor.close()
    connection.close()

    # Your data preprocessing code goes here
    # Training and prediction code goes here

    # Data preprocessing
    trans = ['Start_date', 'End_date']

    for column in trans:
        df[column] = df[column].astype(str)
        df[column] = df[column].str.replace(r'\s+', '', regex=True)
        df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

    df['Start_date'] = pd.to_datetime(df['Start_date'])
    df['End_date'] = pd.to_datetime(df['End_date'])

    argus = df[['Start_date', 'Argus_High']]

    copy_price = argus.copy()
    del argus['Start_date']
    scaler = MinMaxScaler(feature_range=(0, 1))
    argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

    training_size = int(len(argus) * 0.65)
    test_size = len(argus) - training_size
    train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create widgets for hyperparameters
    time_step = st.slider("Time Step", min_value=1, max_value=50, value=10)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.006)
    num_lstm_layers = st.slider("Number of LSTM Layers", min_value=1, max_value=4, value=1)
    lstm_units = st.slider("LSTM Units", min_value=1, max_value=128, value=16)
    batch_size = st.slider("Batch Size", min_value=1, max_value=50, value=10)
    num_epochs = st.slider("Number of Epochs", min_value=1, max_value=100, value=100)

    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(time_step, 1)))
    for _ in range(num_lstm_layers - 1):
        model.add(LSTM(units=lstm_units))
    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)

    loss = model.evaluate(X_test, y_test)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

     # Display the results
    st.subheader("Prediction Results")
    st.write("Test Loss:", loss)

    # Perform prediction for the next week
    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = time_step
    i = 0
    pred_week = 6

    while i < pred_week:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1

    last_week=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_week+1)
    print(last_week)
    print(day_pred)

    temp_mat = np.empty((len(last_week)+pred_week+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]
    Start_date = '2023-07-16'

    last_original_week_value = temp_mat
    next_predicted_week_value = temp_mat

    last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_week_value':last_original_week_value,
        'next_predicted_week_value':next_predicted_week_value
    })

    names = cycle(['Last 15 week close price','Predicted next 10 week price'])
    new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')

    fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                labels={'value': 'Stock price'},
                title='Compare last 15 week vs next 10 week')

    fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()
    
    

    

    # Training and prediction code goes here

    # Display the results
    st.subheader("Prediction Results")
    st.write("Test Loss:", loss)

    # Plot the results using Plotly Express
    fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                  labels={'value': 'Stock price'},
                  title='Compare last 15 week vs next 10 week')

    fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

    st.plotly_chart(fig)
else:
    st.error("Failed to connect to the database. Check your database connection settings.")
