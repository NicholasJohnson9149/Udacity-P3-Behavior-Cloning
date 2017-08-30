#
# 	Behavioral Cloning Project 
# 	July 2017 
#	Author Nicholas Johnson 
#	Software uses Keras to create a Machine Learning Pipesample
#	The Output will be a stearing angel based on training data
#	Goal to Drive a Car around a simulated track 
#

import csv
import cv2 
import numpy as np
import os
import sklearn
from sklearn.utils import shuffle
np.random.seed(0)

def generator(images, measurements, samples, batch_size, is_training):
	#    Generator to process a certain portion of the model at a time
	num_samples = len(samples)

	# images = np.empty([batch_size, height, width, num_channels])
 	# steering = np.empty(batch_size)

	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			measurements = []
			for batch_samples in batch_samples:
				for i in range(3):
					source_path = sample[i]
					name = source_path.split('/')[-1]
					current_path = '../data/IMG/' + name
					image = cv2.imread(current_path)
					images.append(image)
					correction = 0.02 
					# Number was chosen with trail and error, I tried 1 and found the car was jerky.
					# I saw David use 0.02 and felt it was too week at times, so I selected 0.05. 
					# Went with 0.02 after more exirmenting
					measurement = float(sample[3])
					measurements.append(measurement)
					measurements.append(measurement+correction)
					measurements.append(measurement-correction)

			augmented_images = []
			augmented_measurements = []
			for image, measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				flipped_image = cv2.flip(image, 1)
				flipped_measurement = float(measurement) * -1.0
				augmented_images.append(flipped_image)
				augmented_measurements.append(flipped_measurement)			

		# trim image to only see section with road 
		X_data = np.array(augmented_images)
		y_data = np.array(flipped_measurement)
		print("X_train ", X_data.shape)
		print("y_train ", y_data.shape)
		#sklearn.utils.shuffle(X_data, y_data)
		yield sklearn.utils.shuffle(X_data, y_data)

# plot function to be called after running the model
def plot_results(history, num = 0):
    #Plot the results 
    #:param history: The fit model
    #:param num: The number for the output file to save to
    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('training_validation_loss_plot_' + str(num) + '.jpg')
    plt.show()
    plt.close()

#import data 
lines = []
with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in lines:
                lines.append(line)

# images = []
# measurements = []
# for sample in samples:
#         for i in range(3):
#                 source_path = sample[i]
#                 filename = source_path.split('/')[-1]
#                 current_path = '../data7/IMG/' + filename
#                 image = cv2.imread(current_path)
#                 images.append(image)
#         correction = 0.05
#         measurement = float(sample[3])
#         measurements.append(measurement)
#         measurements.append(measurement+correction)
#         measurements.append(measurement-correction)   

# augmented_images = []
# augmented_measurements = []
# for image, measurement in zip(images, measurements):
# 	augmented_images.append(image)
# 	augmented_measurements.append(measurement)
# 	lipped_image = cv2.flip(image, 1)
# 	flipped_measurement = float(measurement) * -1.0
# 	augmented_images.append(flipped_image)
# 	augmented_measurements.append(flipped_measurement)
# 	augmented_measurements.append(flipped_measurement+correction)
# 	augmented_measurements.append(flipped_measurement-correction)

# X_data = np.array(augmented_images)
# y_data = np.array(augmented_measurements)

# split Data into Traning and Validation 
test_size = 0.2
random_state = 0
# compile and train the model using the generator function
from sklearn.model_selection import train_test_split
# train_samples, validation_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
# train_generator = generator(lines, images, measurmnets, train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)

# Make lines a numpy array

# from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(samples, test_size=test_size, random_state=random_state)
X_train_generator = generator(X_train, batch_size=32)
X_validation_generator = generator(X_valid, batch_size=32)
y_train_generator = generator(y_train, batch_size=32)
y_validation_generator = generator(y_valid, batch_size=32)


print('X_train shape', X_train.shape)
print('Y_train shape', y_train.shape)
print('X_valid shape', X_valid.shape)
print('y_valid shape', y_valid.shape)

# import infomation from keras for creating the learning model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint

# model peramiters 
keep_prob = 0.5
video_H = 160
viedo_L = 320
layers = 3
crop_H = 40
crop_W = 80


model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(video_H, viedo_L, layers)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x / 255.0 - 1., input_shape=(video_H, viedo_L, layers), output_shape=(video_H, viedo_L,layers)))
model.add(Cropping2D(cropping=((crop_W, crop_H), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(keep_prob))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))
model.summary()


#Create a checkpoint of the model 
checkpoint = ModelCheckpoint('model-{epoch:03d}.h0',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

# set constant numbers for model generation
batch_size = 40
num_epochs = 10
samples_per_epoch = 20000
learn_rate = 0.0001

# create a model and save it
model.compile(loss='mse', optimizer=adam(lr=learn_rate))
#model.fit(images, steering, validation_split=0.2, shuffle=True, nb_epoch=epoch) - Used for Orginal training 

# history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), callbacks=[checkpoint], nb_epoch=5, verbose=1)

# Create a model using the fit_generator  
history = model.fit_generator(generator(X_train, y_train, batch_size,),
                    samples_per_epoch,
                    num_epochs,
                    max_q_size=1,
                    validation_data=generator(X_valid, y_valid, batch_size),
                    nb_val_samples=len(X_valid),
                    callbacks=[checkpoint],
                    verbose=1)

model.save('nvidia07.h5')
plot_results(history_object, 0)
exit()
