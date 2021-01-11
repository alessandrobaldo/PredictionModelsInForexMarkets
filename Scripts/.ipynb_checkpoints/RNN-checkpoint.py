from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from tensorflow.keras.models import model_from_json
import numpy as np
from .DTW import *
from .FastDTW import *

def create_rnn(nb_layers, sizeCell, num_features):
	model = Sequential()

	# Recurrent layer
	#The heart of the network: a layer of LSTM cells with dropout to prevent overfitting. 
	#Since we are only using one LSTM layer, it does not return the sequences,
	#for using two or more layers, make sure to return sequences.
	
	model.add(LSTM(sizeCell,input_shape=(28,28),return_sequences=True)) 
	model.add(Dropout(0.2))
	
	for i in range(nb_layers-2):
		model.add(LSTM(sizeCell,return_sequences=True))
		model.add(Dropout(0.2))

	# The input to the LSTM layer is (None, features, 100) which means that for each batch (the first dimension), 
	#each sequence has 'features' timesteps (words), each of which has 100 features after embedding. 
	#Input to an LSTM layer always has the (batch_size, timesteps, features) shape.

	#model.add(LSTM(100, return_sequences=True))
	model.add(LSTM(sizeCell, return_sequences=False))
	model.add(Dropout(0.2))

	# Fully connected layer
	#A fully-connected Dense layer with relu activation. This adds additional representational capacity to the network.
	#model.add(Dense(64, activation='relu'))

	# Dropout for regularization
	#A Dropout layer to prevent overfitting to the training data.
	#model.add(Dropout(0.5))

	# Output layer
	#A Dense fully-connected output layer. This produces a probability for every word in the vocab using softmax activation.
	model.add(Dense(1, activation='linear'))
	return model


#KERAS CALLBACKS: ModelCheckpoint and EarlyStopping to avoid overfitting during training
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def run(train_X, valid_X, test_X, train_Y, valid_Y,test_Y, epochs, batch_size,learning_rate, nb_layers, sizeCell, num_features, dataset):
	model = create_rnn(nb_layers, sizeCell,num_features)
	
	# Compile the model
	opt = Adam(lr=learning_rate, decay=1e-3 /200)
	model.compile(loss="mean_squared_error", optimizer=opt)
	
	# Create callbacks
	callbacks = [EarlyStopping(monitor='val_loss', patience=np.max([epochs/2, 30]), restore_best_weights=True)]
	
	callbacks.append(ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=int(epochs/5), verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0.0000001))
	
	
	print("[INFO] training model...")
	history = model.fit(train_X,  train_Y, 
						batch_size=batch_size, epochs=epochs,
						callbacks=callbacks,
						validation_data=(valid_X, valid_Y), verbose = 2)

	saveModel(model,dataset)
	
	print("[INFO] predicting currency prices...")
	preds = model.predict(test_X)
	preds = np.reshape(preds,len(preds))
	mse = np.mean((preds - test_Y)**2)
	corr = np.corrcoef(preds, test_Y)[0,1]
	dti = DTW(preds, test_Y, 1)
	fast_dti = fastdtw(preds, test_Y, 1)
	
	return preds, mse, corr, dti, fast_dti


def saveModel(model,dataset):
	model_json = model.to_json()
	with open("Models/RNN/modelRNN{}.json".format(dataset), "w+") as json_file:
		json_file.write(model_json)

	model.save_weights("Models/RNN/modelRNN{}.h5".format(dataset))
	  
	

def runLoadedModel(test_X,test_Y, dataset):
	# load json and create model
	json_file = open('Models/RNN/modelRNN{}.json'.format(dataset), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('Models/RNN/modelRNN{}.h5'.format(dataset))
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
