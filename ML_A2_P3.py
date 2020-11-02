import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == "__main__":

	batch_size = 128
	num_classes = 5
	epochs = 12

	# input image dimensions
	img_rows, img_cols = 28, 28

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

	train_subset_idx = np.where(y_train < num_classes)[0]
	test_subset_idx = np.where(y_test < num_classes)[0]

	x_train, y_train, x_test, y_test = x_train[train_subset_idx], y_train[train_subset_idx], x_test[test_subset_idx], \
	                                   y_test[test_subset_idx]

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = keras.Sequential([
		keras.layers.Flatten(),
		keras.layers.Dense(32, activation='relu',input_shape=(28, 28, 1)),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(64, activation='relu'),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(num_classes, activation='softmax')
	])

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer='adam', metrics=['accuracy',tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test))

	score = model.evaluate(x_test, y_test, verbose=0)
	print(model.metrics_names)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])s
	print('Test precision:', score[2])
	print('Test recall:', score[3])

