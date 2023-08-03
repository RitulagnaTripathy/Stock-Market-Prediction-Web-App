import pandas as pd
import numpy as np
from plotly import graph_objs as go

from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0] 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def lstm_model(data):
    pass
    closedf = data.reset_index()['Close']

    #Feature Scaling
    sc = MinMaxScaler(feature_range=(0,1))
    closedf = sc.fit_transform(np.array(closedf).reshape(-1,1))

    #Splitting dataset into training and test set
    training_size=int(len(closedf)*0.8)
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:]

    #Creating time steps
    time_step = 100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    #Reshaping input to be [samples, time steps, features] as LSTM requires 3-D input
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    #Building a sequential model
    keras.backend.clear_session()
    model = Sequential()
    #Number of hidden units is considered as 50
    model.add(LSTM(50, return_sequences = True, input_shape = (time_step,1))) #Shape of the input data that is fed to the LSTM network
    model.add(LSTM(50, return_sequences = True)) 
    model.add(LSTM(50)) 
    model.add(Dense(1)) #A dense layer is a fully-connected layer, i.e. every neurons of the layer N are connected to every neurons of the layer N+1.
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=10, batch_size=75, verbose=1)

    #Prediction
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    #Transform back to original form
    train_predict = sc.inverse_transform(train_predict)
    test_predict = sc.inverse_transform(test_predict)
    Y_train = sc.inverse_transform(Y_train.reshape(-1,1))
    Y_test = sc.inverse_transform(Y_test.reshape(-1,1))

    #Comparison between Actual Close Price vs Predicted Close Price
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[time_step:len(train_predict)+time_step, :] = train_predict

    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(time_step*2)+1:len(closedf)-1, :] = test_predict

    #Predicting stock price for the next 30 days
    #Getting the last 100 days records
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

    #Creating a list of last 100 days data
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    #Predicting next 30 days price using current data

    output=[]
    i=0
    while(i<30):
        if (len(temp_input)>time_step):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            y = model.predict(np.expand_dims(x_input, 2))
            temp_input.extend(y[0])
            temp_input=temp_input[1:]
            output.extend(y.tolist())
            i=i+1
            
        else:
            y = model.predict(np.expand_dims(x_input, 2))
            temp_input.extend(y[0])
            output.extend(y.tolist())   
            i=i+1

    output = sc.inverse_transform(np.array(output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    closedf = sc.inverse_transform(np.array(closedf[-100:]).reshape(-1,1)).reshape(1,-1).tolist()[0]

    return trainPredictPlot, testPredictPlot, output, closedf

