import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe 
#tf.enable_eager_execution()
from tensorflow.keras.models import model_from_json

layers = tf.keras.layers

class TemporalBlock(tf.keras.Model):
	def __init__(self, dilation_rate, nb_filters, kernel_size, 
					   padding, dropout_rate=0.0): 
		super(TemporalBlock, self).__init__()
		init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
		assert padding in ['causal', 'same']

		# block1

		self.conv1 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
								   dilation_rate=dilation_rate, padding=padding, 
								   kernel_initializer=init, input_shape = (None, 784))
		self.batch1 = layers.BatchNormalization(axis=-1)
		self.ac1 = layers.Activation('relu')
		self.drop1 = layers.Dropout(rate=dropout_rate)
		
		# block2
		self.conv2 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
								   dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
		
		self.batch2 = layers.BatchNormalization(axis=-1)		
		self.ac2 = layers.Activation('relu')
		self.drop2 = layers.Dropout(rate=dropout_rate)

		
		self.downsample = layers.Conv1D(filters=nb_filters, kernel_size=1, 
										padding='same', kernel_initializer=init)
		self.ac3 = layers.Activation('relu')


	def call(self, x, training):
		prev_x = x
		x = self.conv1(x)
		x = self.batch1(x)
		x = self.ac1(x)
		x = self.drop1(x) if training else x

		x = self.conv2(x)
		x = self.batch2(x)
		x = self.ac2(x)
		x = self.drop2(x) if training else x

		if prev_x.shape[-1] != x.shape[-1]:    # match the dimension
			prev_x = self.downsample(prev_x)
		assert prev_x.shape == x.shape

		return self.ac3(prev_x + x)            # skip connection

# test:
# x = tf.convert_to_tensor(np.random.random((100, 10, 50)))   # batch, seq_len, dim
# model = TemporalBlock(dilation_rate=3, nb_filters=70, kernel_size=3, padding='causal', dropout_rate=0.1)
# y = model(x)
# print(y.shape)


class TemporalConvNet(tf.keras.Model):
	def __init__(self, num_channels, kernel_size=2, dropout=0.2):
		# num_channels is a list contains hidden sizes of Conv1D
		super(TemporalConvNet, self).__init__()
		assert isinstance(num_channels, list)

		model = tf.keras.Sequential()

		# The model contains "num_levels" TemporalBlock
		num_levels = len(num_channels)
		for i in range(num_levels):
			dilation_rate = 2 ** i                  # exponential growth
			model.add(TemporalBlock(dilation_rate, num_channels[i], kernel_size, 
					  padding='causal', dropout_rate=dropout))
		self.network = model

	def call(self, x, training):
		return self.network(x, training=training)

# test:
# x = tf.convert_to_tensor(np.random.random((100, 10, 50)))   # batch, seq_len, dim
# model = TemporalConvNet(num_channels=[70, 80])
# y = model(x)
# print(y.shape)


class TCN(tf.keras.Model):
	def __init__(self, output_size, num_channels, kernel_size, dropout):
		super(TCN, self).__init__()
		init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

		self.temporalCN = TemporalConvNet(num_channels, kernel_size=kernel_size, dropout=dropout)
		self.linear = tf.keras.layers.Dense(output_size, kernel_initializer=init)

	def call(self, x, training=True):
		y = self.temporalCN(x, training=training)
		return self.linear(y[:, -1, :])   # use the last element to output the result



def run(train_X, valid_X, test_X, train_Y, valid_Y, test_Y, epochs, batch_size,lr, num_channel, nb_filter, mse_min, corr_max):
	model = TCN(output_size = 1,num_channels = [nb_filter]*num_channel,kernel_size = 3,dropout = 0.2)

	#train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X)).batch(batch_size)
	train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size)
	valid_data, valid_labels = tf.convert_to_tensor(valid_X), tf.convert_to_tensor(valid_Y)
	
	optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
	#mse = tf.keras.losses.MeanSquaredError()
	# run 
	for epoch in range(epochs):
		for batch, (train_x, train_y) in enumerate(train_dataset):
			# loss
			with tf.GradientTape() as tape:
				y = model(train_x, training=True)
				#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_Y, logits=y))
				loss = tf.reduce_mean(tf.keras.losses.mse(train_y, y))
				if epoch == 1:
					loss_prec = loss
				else:
					if loss>loss_prec:
						lr = lr/2
						optimizer = tf.keras.optimizers.Adam(learning_rate=lr)	
			# gradient
			gradient = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(gradient, model.trainable_variables))
			#print("Batch:", batch, ", Train loss:", loss.numpy())

		# Eval Acc
		eval_labels =  model(valid_data, training=False)
		eval_mse = tf.reduce_mean(tf.keras.losses.mse(eval_labels.numpy(), valid_labels.numpy())).numpy()
		print("Epoch:", epoch, ", Eval mse:", eval_mse)
		
	
	test_data, test_labels = tf.convert_to_tensor(test_X), tf.convert_to_tensor(test_Y)
	predicted_labels = model(test_data, training=False)
	mse = tf.reduce_mean(tf.keras.losses.mse(predicted_labels.numpy(), test_labels.numpy())).numpy()
	
	if mse < mse_min:
		model_json = model.to_json()
		with open("Models/modelTCN.json", "w+") as json_file:
			json_file.write(model_json)
		
		model.save_weights("Models/modelTCN.h5")
		with open("Models/BestTCN.txt","w+") as bestFile:
			bestFile.write("Epochs: {}\nBatch Size: {}\nLearning Rate: {}Number of Channels:{}\nNumber of Filters:{}\nKernelSize: {}\nAdaptive Learning Rate: Yes".format(epochs, batch_size,learning_rate, num_channel, nb_filter, 3))
		
	return predicted_labels, mse
