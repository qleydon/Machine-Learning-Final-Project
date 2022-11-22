import numpy as np
from math import sqrt
from matplotlib import pyplot
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

# load dataset
dataset = read_csv('pollution_quinn.csv', header=0, index_col=0)
#dataset.drop('wnd_dir', axis=1, inplace=True)
values = dataset.values
y_temp = np.zeros((values.shape[0],1))
cnt_1 = 0
# Convert direction to numerical value.
for i in range(values.shape[0]):
    if values[i,4] == 'N':
        values[i, 4] =1
    elif values[i,4] == 'NE':
        values[i, 4] =0.125
    elif values[i,4] == 'E':
        values[i, 4] =0.25
    elif values[i,4] == 'SE':
        values[i, 4] =0.375
    elif values[i,4] == 'S':
        values[i, 4] =0.5
    elif values[i,4] == 'SW':
        values[i, 4] =0.625
    elif values[i,4] == 'W':
        values[i, 4] =0.75
    elif values[i, 4] == 'NW':
        values[i, 4] = 0.875
    else:
        values[i,4] = 0

    if values[i,0] >= 250.5:
        y_temp[i,0] = 1
        cnt_1 = cnt_1+1
    else:
        y_temp[i,0] = 0
print(cnt_1)
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group], color='green')
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

n_train_hours = 365 * 24 * 2

offset = 3

values = values.astype('float32')
x_mean = np.mean(values, axis=0)
x_std = np.std(values, axis=0)
values = (values - x_mean) / x_std

#x_max = np.max(values, axis=0)
#x_min = np.min(values, axis=0)
#values = (values-x_min)/(x_max - x_min)

print(values[:5, :])

train = int(values.shape[0] * 0.6)
val = int(values.shape[0] * 0.8)

train_X1 = values[:train, :]
train_y = y_temp[offset:train+offset, 0]

val_X1 = values[train:val, :]
val_y = y_temp[(train+offset):((val)+offset), 0]

test_X1 = values[(val):-offset, :]
test_y = y_temp[(val)+offset:, 0]


train_X = train_X1.reshape(train_X1.shape[0], 1, train_X1.shape[1])
val_X = val_X1.reshape(val_X1.shape[0], 1, val_X1.shape[1])
test_X = test_X1.reshape(test_X1.shape[0], 1, test_X1.shape[1])


# design network
model = Sequential()
model.add(LSTM(10, input_shape=(1, 8)))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(val_X, val_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
#yhat = yhat * x_std[0] + x_mean[0]
yhat[yhat >= 0.5] = 1
yhat[yhat < 0.5] = 0

y = test_y.reshape((test_y.shape[0], 1))
#y = y * x_std[0] + x_mean[0]
#y = y * (x_max[0] - x_min[0])+x_min[0]
# calculate RMSE
rmse_LSTM = sqrt(mean_squared_error(y, yhat))
print('Test RMSE: %.3f' % rmse_LSTM)

pyplot.plot(y, 'c-', yhat, 'm-')
pyplot.legend(['True Y', 'Predicted Y'])
pyplot.title('LSTM Predicted Pollution')
pyplot.ylabel('Pollution')
pyplot.xlabel('Hour')
pyplot.show()

y_lstm = yhat

print('Test Accuracy: %.4f' %accuracy_score(y, yhat))
print('Test Precision: %.4f' %precision_score(y, yhat))
print('Test Recall: %.4f' %recall_score(y, yhat))
cf_matrix = confusion_matrix(y, yhat)

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('LSTM Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Acceptable','Hazardous'])
ax.yaxis.set_ticklabels(['Acceptable','Hazardous'])

## Display the visualization of the Confusion Matrix.
pyplot.show()
# ---------------------------------------------------------------------------------------------------
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D

# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
train_X_conv = train_X.reshape(train_X1.shape[0], 2, 4, 1)
val_X_conv = val_X.reshape(val_X1.shape[0], 2, 4, 1)
test_X_conv = test_X.reshape(test_X1.shape[0], 2, 4, 1)

# design network
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'), input_shape=(None, 4, 1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X_conv, train_y, epochs=50, batch_size=72, validation_data=(test_X_conv, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
model.summary()

# make a prediction
yhat = model.predict(test_X_conv)
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
yhat[yhat >= 0.5] = 1
yhat[yhat < 0.5] = 0

y = test_y.reshape((test_y.shape[0], 1))

# calculate RMSE
rmse_conv = sqrt(mean_squared_error(y, yhat))
print('Test RMSE: %.3f' % rmse_conv)

pyplot.plot(y, 'c-', yhat, 'm-')
pyplot.legend(['True', 'Predicted'])
pyplot.title('CNN-LSTM Predicted Pollution')
pyplot.ylabel('Pollution')
pyplot.xlabel('Hour')
pyplot.show()

print(f"LSTM RMSE: {rmse_LSTM}      CONV_LSTM RSME: {rmse_conv}")


print('Test Accuracy: %.4f' %accuracy_score(y, yhat))
print('Test Precision: %.4f' %precision_score(y, yhat))
print('Test Recall: %.4f' %recall_score(y, yhat))
cf_matrix = confusion_matrix(y, yhat)

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('CNN-LSTM Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Acceptable','Hazardous'])
ax.yaxis.set_ticklabels(['Acceptable','Hazardous'])

## Display the visualization of the Confusion Matrix.
pyplot.show()