##
# Behavioral Cloning Project 
# 	July 2017 
#	Author Nicholas Johnson 
#	Software uses Keras to create a Machine Learning Pipeline
#	The Output will be a stearing angel based on training data
#	Goal to Drive a Car around a simulated track 
#

import csv
import cv2nvidia 
import numpy as np
import os
import sklearn
from utils import *
np.random.seed(0)

lines = []
with open('../data6/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
                lines.append(line)

images = []
measurements = []
for line in lines:
        for i in range(3):
                source_path = line[i]
                filename = source_path.split('/')[-1]
                current_path = '../data6/IMG/' + filename
                image = cv2.imread(current_path)
                images.append(image)
        correction = 0.05
        measurement = float(line[3])
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
        augmented_measurements.append(flipped_measurement+correction)
        augmented_measurements.append(flipped_measurement-correction)

X_data = np.array(images)
y_data = np.array(measurements)

print('X_data:', X_data.shape)
print('y_data:', y_data.shape)

#Split Data into Traning and Validation 
test_size = 0.2
random_state = 0

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
print('X_train shape', X_train.shape)
print('Y_train shape', y_train.shape)
print('X_valid shape', X_valid.shape)
print('y_valid shape', y_valid.shape)

def generator(data_dir, image_paths, steer_angles, batch_size, is_training):
	#    Generator to process a certain portion of the model at a time
	#    :param data_dir: The data directory
	#    :param image_paths: The paths to the images
	#    :param steer_angles: The steering angles
	#    :param batch_size: The batch size
	#    :param is_training: Whether this is training data (True) or validation data (False)
	
	images = np.empty([batch_size, height, width, num_channels])
	steering = np.empty(batch_size)
    
	while True:
	    i = 0
	    for index in np.random.permutation(image_paths.shape[0]):
	        center, left, right = image_paths[index]
	        steering_angle = steer_angles[index]

	        if is_training and np.random.rand() < 0.6:
	            image, steering_angle = augment_image(data_dir, center, left, right, steering_angle)
	        else:
	            image = load_image(data_dir, center) 
	             
	        image = preprocess(image)
	            
	        images[i] = image
	        steering[i] = steering_angle

	        i += 1
	        if i == batch_size:
	            break
	                
	    yield images, steering
        
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#model peramiters 
keep_prob = 0.5
video_H = 160
viedo_L = 320
layers = 3
crop_H = 25
crop_W = 70

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(video_H, viedo_L, layers)))
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

# Set Number of epochs Training 
epoch = 10

model.compile(loss='mse', optimizer='adam')
model.fit(images, steering, validation_split=0.2, shuffle=True, nb_epoch=epoch)

model.save('nvidia01.h5')

history = model.fit_generator(generator(X_train, y_train, batch_size, True),
                    samples_per_epoch,
                    epoch,
                    max_q_size=1,
                    validation_data=generator(img_path, X_valid, y_valid, batch_size, False),
                    nb_val_samples=len(X_valid),
                    callbacks=[checkpoint],
                    verbose=1)

plot_results(history, 0)
exit()
