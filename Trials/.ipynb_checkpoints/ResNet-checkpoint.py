from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from tensorflow.keras.models import model_from_json


def run(train_X,valid_X, test_X, train_Y, valid_Y,test_Y, epochs,batch_size, lr, acc_max):
	input = Input(shape = train_X.shape[1:])
	resnet = ResNet50V2(weights=None, include_top=False, pooling='max')(input)
	flattened = Flatten()(resnet) 
	fc1 = Dense(256, activation = 'relu')(flattened)
	output = Dense(2, activation = 'softmax')(fc1)
	
	model = Model(input, output)
	
	opt = Adam(lr=lr, decay=1e-3 )
	model.compile(loss=BinaryCrossentropy(), optimizer=opt)
	
	print("[INFO] training model...")
	
	# Create callbacks# Create callbacks
	callbacks = [EarlyStopping(monitor='val_loss', patience=np.max([epochs/2, 30]), restore_best_weights=True)]
	
	callbacks.append(ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=int(epochs/5), verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0.0000001))
	
	model.fit(x=train_X, y=train_Y, 
		validation_data=(valid_X, valid_Y),
		epochs=epochs, batch_size=batch_size, callbacks = callbacks, verbose = 2)

	
	print("[INFO] testing model...")
	preds = model.predict(test_X)
	#preds = preds.flatten()
	
	
	acc = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y)])/len(test_Y)
	acc0 = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y) if true == 0])/len(test_Y[test_Y == 0])
	acc1 = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y) if true == 1])/len(test_Y[test_Y == 1])

	if acc*acc1 > acc_max:
		model_json = model.to_json()
		with open("resNet.json", "w+") as json_file:
			json_file.write(model_json)
			
		model.save_weights("resNet.h5")
		with open("resNet.txt","w+") as bestFile:
			bestFile.write("Epochs: {}\nBatch Size: {}\nLearning Rate: {}".format(epochs, batch_size,lr))
	
		
	return preds,acc, acc0, acc1

def runLoadedModel(test_X,test_Y):
	# load json and create model
	json_file = open('resNet.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('resNet.h5')
	#loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	#score = loaded_model.evaluate(test_X,test_Y, verbose=0)
	preds = loaded_model.predict(test_X)
	acc = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y)])/len(test_Y)
	acc0 = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y) if true == 0])/len(test_Y[test_Y == 0])
	acc1 = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y) if true == 1])/len(test_Y[test_Y == 1])
	print("Accuracy: {}".format(acc))
	print("Accuracy on Pivot Points: {}".format(acc1))
	print("Accuracy on non-Pivot Points: {}".format(acc0))
	return preds
	