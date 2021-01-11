from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from tensorflow.keras.models import model_from_json
from TemporalLoss import *


def create_cnn(shape, kernel_size, filters=(16, 32, 64)):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = shape
	chanDim = -1
	
	# define the model input
	inputs = Input(shape=inputShape)
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
		# CONV => RELU => BN => POOL
		x = Conv2D(f, (kernel_size, kernel_size), padding="same")(x) #kernel kernel_size x kernel_size x f, so it produces f images
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
	
	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x) #FC (Fully-Connected) Layer
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)
	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(4)(x)
	x = Activation("relu")(x)
	# check to see if the regression node should be added
	x = Dense(1, activation="linear")(x)
	# construct the CNN
	model = Model(inputs, x)
	# return the CNN
	return model

def run(train_X,valid_X, test_X, train_Y, valid_Y,test_Y, epochs, batch_size, filters,kernel, lr,best_loss):
	# create our Convolutional Neural Network and then compile the model
	# using mean absolute percentage error as our loss, implying that we
	# seek to minimize the absolute percentage difference between our
	# price *predictions* and the *actual prices*
	model = create_cnn(train_X.shape[1:], kernel, filters)
	opt = Adam(lr=lr, decay=1e-3 )
	stl = SpatialTemporalLoss(batch_size)
	model.compile(loss=stl.spatialTemporalLoss, optimizer=opt)
	
	# train the model
	print("[INFO] training model...")
	
	# Create callbacks# Create callbacks
	callbacks = [EarlyStopping(monitor='val_loss', patience=np.max([epochs/2, 30]), restore_best_weights=True)]
	
	callbacks.append(ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=int(epochs/5), verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0.0000001))
	
	model.fit(x=train_X, y=train_Y, 
		validation_data=(valid_X, valid_Y),
		epochs=epochs, batch_size=batch_size, callbacks = callbacks, verbose = 2)

	
	preds = model.predict(test_X)
	preds = preds.flatten()
	
	loss = stl.spatialTemporalLoss(preds, test_Y)
	if loss < best_loss: #check if it's nan	
		model_json = model.to_json()
		with open("modelCNN.json", "w+") as json_file:
			json_file.write(model_json)

		model.save_weights("modelCNN.h5")
		with open("BestCNN.txt","w+") as bestFile:
			bestFile.write("Epochs: {}\nBatch Size: {}\nLearning Rate: {}\nFilters:{}\nKernel Size:{}".format(epochs, batch_size,lr,filters, kernel))
	
		
	return preds, loss

def runLoadedModel(test_X,test_Y):
	# load json and create model
	json_file = open('modelCNN.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('modelCNN.h5')
	#loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	#score = loaded_model.evaluate(test_X,test_Y, verbose=0)
	preds = loaded_model.predict(test_X)
	print("SpatialTemporalLoss: {}".format(spatialTemporalLoss(preds, test_Y)))
	return preds


