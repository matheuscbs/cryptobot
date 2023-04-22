import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Adicione estas funções à classe EthereumForecast
def create_dataset(self, data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def run_lstm(self, epochs=100, look_back=1):
    data = self.historical_data[:, 1].reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size, :], data[train_size:len(data), :]

    look_back = look_back
    trainX, trainY = self.create_dataset(train, look_back)
    testX, testY = self.create_dataset(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(4, input_shape=(1, look_back)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print(f'Train Score: {trainScore:.2f} RMSE')
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print(f'Test Score: {testScore:.2f} RMSE')

    return testPredict, testY
