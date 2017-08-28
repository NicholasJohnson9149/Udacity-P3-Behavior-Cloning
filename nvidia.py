##
# Behavioral Cloning Project 
# 	July 2017 
#	Author Nicholas Johnson 
#	Software uses Keras to create a Machine Learning Pipeline
#	The Output will be a stearing angel based on training data
#	Goal to Drive a Car around a simulated track 
#

import csv
import cv2
import numpy as np
import os
import sklearn

lines = []
with open('../data6/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
                lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

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

X_train = np.array(images)
y_train = np.array(measurements)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

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

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)model = Sequential()
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epoch)

model.save('nvidia01.h5')
exit()
