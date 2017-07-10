import load_data
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import CSVLogger

PATCH_SIZE = (1, 1)

# Create name for model file based on script filename
FILENAME_SCRIPT = os.path.basename(__file__)
FILENAME = FILENAME_SCRIPT[:len(FILENAME_SCRIPT)-3]
FILENAME_MODEL = FILENAME + '.h5'
FILENAME_LOG = FILENAME + '.log'

# Load data
(X_train, y_train), (X_test, y_test) = load_data.load_data()


# Define model

# Create the model
model = Sequential()

# Input
# 0
model.add(Convolution2D(
	128, 
	(1, 1), 
	input_shape=(1,1,230), 
	activation='relu', 
	border_mode='same'))
	
model.add(BatchNormalization())

# 1
model.add(Convolution2D(
	512, 
	(1, 1), 
	activation='relu', 
	border_mode='same'))

# Fully connected
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(9, activation='softmax'))

# Compile model
runs = 500
epochs = 1
lrate = 0.01
decay = lrate/3
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(
	loss='categorical_crossentropy', 
	optimizer=sgd, 
	metrics=['accuracy'])
	
print(model.summary())

# Fit the model
np.random.seed(load_data.seed)

if os.path.isfile(FILENAME_MODEL):
	model = load_model(FILENAME_MODEL)

logger = CSVLogger(FILENAME_LOG, separator=',', append=True)

for i in range(runs):
	model.fit(
		X_train, 
		y_train, 
		validation_data=(X_test, y_test), 
		nb_epoch=epochs, 
		batch_size=1024,
		callbacks=[logger])

	model.save(FILENAME_MODEL)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
