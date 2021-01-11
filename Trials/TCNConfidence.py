import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Activation, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
tf.random.set_seed(42)

def create_tcn(num_channels,output_size = 28,kernel_size = 3,dropout = 0.2):
	
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

def run(train_X, valid_X, test_X, train_Y, valid_Y, test_Y, epochs, batch_size,lr, num_channel, nb_filter, mse_min, corr_max):
	model = create_tcn([nb_filter]*num_channel)
	
	opt = Adam(learning_rate=lr)
	model.compile(loss="mse", optimizer=opt)
	model.fit(train_X,  train_Y, batch_size=batch_size, epochs=epochs,validation_data=(valid_X, valid_Y), verbose = 2)
  
	preds = model.predict(test_X)
	mse = ((preds - test_Y)**2).mean(axis=None)
	corr = np.mean([np.corrcoef(preds[i,:], test_Y[i,:])[0,1] for i in range(preds.shape[0])])
	
	if corr == corr and corr>0: #check if it's nan
		factor = np.sqrt(mse) * (1-corr)
		best_factor = np.sqrt(mse_min) * (1-corr_max)
		if factor < best_factor:
			model_json = model.to_json()
			with open("Models/modelTCNconfidence.json", "w+") as json_file:
				json_file.write(model_json)

			model.save_weights("Models/modelTCNconfidence.h5")
			with open("Models/BestTCNconfidence.txt","w+") as bestFile:
				bestFile.write("Epochs: {}\nBatch Size: {}\nLearning Rate: {}Number of Channels:{}\nNumber of Filters:{}\nKernelSize: {}\nAdaptive Learning Rate: Yes".format(epochs, batch_size,lr, num_channel, nb_filter, 3))
	return preds, mse, corr

def runLoadedModel(test_X,test_Y=None):
	# load json and create model
	json_file = open('Models/modelTCNconfidence.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('Models/modelTCNconfidence.h5')
	#loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	#score = loaded_model.evaluate(test_X,test_Y, verbose=0)
	preds = loaded_model.predict(test_X)
	if test_Y is not None:
		mse = np.mean((preds - test_Y)**2)
		print("MSE: {}".format(mse))
		corr = np.mean([np.corrcoef(preds[i,:], test_Y[i,:])[0,1] for i in range(preds.shape[0])])
		print("CORR: {}".format(corr))
	return preds
