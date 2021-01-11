import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Activation, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from .DTW import *
from .FastDTW import *
tf.random.set_seed(42)

def create_tcn(num_channels,output_size = 1,kernel_size = 3,dropout = 0.2):
	
	init = RandomNormal(mean=0.0, stddev=0.01, seed = 42)
	
	num_levels = len(num_channels)
	inputs = Input(shape=(784,1))
	x = inputs
	#temporal ConvNet
	for i in range(num_levels):
		#temporal Block
		dilation_rate = 2 ** i
		prev_x = x
		#block 1
		x = Conv1D(filters=num_channels[i], kernel_size=kernel_size,dilation_rate=dilation_rate,
				   padding='causal', kernel_initializer=init)(x)
		x = BatchNormalization(axis=-1)(x)
		x = Activation('relu')(x)
		x = Dropout(rate=dropout)(x)
		
		#block 2
		x = Conv1D(filters=num_channels[i], kernel_size=kernel_size,dilation_rate=dilation_rate,
				   padding='causal', kernel_initializer=init)(x)
		
		x = BatchNormalization(axis=-1)(x)
		x = Activation('relu')(x)
		x = Dropout(rate=dropout)(x)
		
		if prev_x.shape[-1] != x.shape[-1]: 
			prev_x = Conv1D(filters=num_channels[i], kernel_size=1, padding='same', kernel_initializer=init)(prev_x)
			
		x = Activation('relu')(prev_x + x)

	#output layer
	x = x[:,-1,:]
	x = Dense(output_size, kernel_initializer=init)(x)
	model = Model(inputs, x)
	return model

def run(train_X, valid_X, test_X, train_Y, valid_Y, test_Y, epochs, batch_size,lr, num_channel, nb_filter, dataset):
	model = create_tcn([nb_filter]*num_channel)
	
	opt = Adam(learning_rate=lr)
	model.compile(loss="mse", optimizer=opt)

	callbacks = [EarlyStopping(monitor='val_loss', patience=np.max([epochs/2, 30]), restore_best_weights=True)]
	
	callbacks.append(ReduceLROnPlateau(
	monitor='val_loss', factor=0.2, patience=int(epochs/5), verbose=1, mode='auto',
	min_delta=0.0001, cooldown=0, min_lr=0.0000001))

	print("[INFO] training model...")
	model.fit(train_X,train_Y, 
				batch_size=batch_size, epochs=epochs,
				callbacks = callbacks,
				validation_data=(valid_X, valid_Y), verbose = 2)

	saveModel(model,dataset)

	print("[INFO] predicting currency prices...")
	preds = model.predict(test_X)
	preds = np.reshape(preds, len(preds))
	mse = np.mean((preds - test_Y)**2)
	corr = np.corrcoef(preds, test_Y)[0,1]
	dti = DTW(preds, test_Y, 1)
	fast_dti = fastdtw(preds, test_Y, 1)
		
	return preds, mse, corr, dti, fast_dti
  

def saveModel(model,dataset):
	model_json = model.to_json()
	with open("Models/TCN/modelTCN{}.json".format(dataset), "w+") as json_file:
		json_file.write(model_json)

	model.save_weights("Models/TCN/modelTCN{}.h5".format(dataset))
	  
	

def runLoadedModel(test_X,test_Y, dataset):
	# load json and create model
	json_file = open('Models/TCN/modelTCN{}.json'.format(dataset), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('Models/TCN/modelTCN{}.h5'.format(dataset))
	print(loaded_model.summary())
	#loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	#score = loaded_model.evaluate(test_X,test_Y, verbose=0)
	preds = loaded_model.predict(test_X)
	preds = np.reshape(preds, len(preds))
	mse = np.mean((preds - test_Y)**2)
	corr = np.corrcoef(preds, test_Y)[0,1]
	dti = DTW(preds, test_Y, 1)
	fast_dti = fastdtw(preds, test_Y, 1)

	print("MSE: {:e}".format(mse))
	print("CORR: {:.3f}".format(corr))
	print("DTW: {:.3f}".format(dti))
	print("FAST DTW: {:.3f}".format(fast_dti[0]))
	return preds, mse, corr, dti, fast_dti
