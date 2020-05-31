from numpy import loadtxt, sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

# load the dataset
dataset = loadtxt('2011-2013.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,1:7]
y = dataset[:,0]
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(y.reshape(-1,1))
X = scalarX.transform(X)
y = scalarY.transform(y.reshape(-1,1))

# define the keras model
model = Sequential()
model.add(Dense(6, input_dim=6, activation='relu'))
model.add(Dense(3, activation='relu'))
# model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='relu'))

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam')

# fit the keras model on the dataset
history = model.fit(X, y, epochs=200, batch_size=8, verbose=0)

# # evaluate the keras model
# _, accuracy = model.evaluate(X, y, verbose=0)
# print('Accuracy: %.2f' % (accuracy*100))

error = model.evaluate(X, y, verbose=0)
print('mse: %.3f, rmse: %.3f' % (error, sqrt(error)))

dataset2 = loadtxt('2014.csv', delimiter=',')
Xnew = dataset2[:,1:7]
# print(Xnew)
Xnew2 = scalarX.transform(Xnew)
# print(Xnew2)
ynew2 = model.predict(Xnew2)
ynew = scalarY.inverse_transform(ynew2)
error2 = model.evaluate(Xnew, ynew, verbose=0)
print('mse: %.3f, rmse: %.3f' % (error2, sqrt(error2)))

ynew3 = ynew.reshape(-1)

pyplot.plot(history.history['loss'], label='train')
pyplot.show()

filename = 'forecast.txt'
with open(filename, "w") as f:
	for i in range(len(ynew3)):
		f.write("{:.2f}\n".format(ynew3[i]))