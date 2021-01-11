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
from .DTW import *
from .FastDTW import *



def create_cnn(width, height, depth, kernel_size, filters=(16, 32, 64)):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
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

def run(train_X,valid_X, test_X, train_Y, valid_Y,test_Y, epochs, batch_size, filters,kernel, lr, dataset):
	# create our Convolutional Neural Network and then compile the model
	# using mean absolute percentage error as our loss, implying that we
	# seek to minimize the absolute percentage difference between our
	# price *predictions* and the *actual prices*
	model = create_cnn(28, 28, 1, kernel, filters)
	opt = Adam(lr=lr, decay=1e-3 )
	model.compile(loss="mean_squared_error", optimizer=opt)
	
	# train the model
	print("[INFO] training model...")
	
	# Create callbacks# Create callbacks
	callbacks = [EarlyStopping(monitor='val_loss', patience=np.max([epochs/2, 30]), restore_best_weights=True)]
	
	callbacks.append(ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=int(epochs/5), verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0.0000001))
	
	model.fit(train_X,train_Y, 
				batch_size=batch_size, epochs=epochs,
				callbacks = callbacks,
				validation_data=(valid_X, valid_Y), verbose = 2)

	saveModel(model,dataset)
	
	preds = model.predict(test_X)
	preds = preds.flatten()
	
	mse = np.mean((preds - test_Y)**2)
	corr = np.corrcoef(preds,test_Y)[0,1]
	dti = DTW(preds, test_Y, 1)
	fast_dti = fastdtw(preds, test_Y, 1)
		
	return preds, mse, corr, dti, fast_dti

def saveModel(model,dataset):
	model_json = model.to_json()
	with open("Models/CNN/modelCNN{}.json".format(dataset), "w+") as json_file:
		json_file.write(model_json)

	model.save_weights("Models/CNN/modelCNN{}.h5".format(dataset))
	  
	

def runLoadedModel(test_X,test_Y, dataset):
	# load json and create model
	json_file = open('Models/CNN/modelCNN{}.json'.format(dataset), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('Models/CNN/modelCNN{}.h5'.format(dataset))
	print(loaded_model.summary())
	#loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	#score = loaded_model.evaluate(test_X,test_Y, verbose=0)
	preds = loaded_model.predict(test_X)
	preds = preds.flatten()
	mse = np.mean((preds - test_Y)**2)
	corr = np.corrcoef(preds, test_Y)[0,1]
	dti = DTW(preds, test_Y, 1)
	fast_dti = fastdtw(preds, test_Y, 1)

	print("MSE: {:e}".format(mse))
	print("CORR: {:.3f}".format(corr))
	print("DTW: {:.3f}".format(dti))
	print("FAST DTW: {:.3f}".format(fast_dti[0]))
	return preds, mse, corr, dti, fast_dti