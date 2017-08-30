
import csv
import cv2
import numpy as np
import os
import sklearn

lines = []
with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
                lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
    correction = 0.05 
    # Number was chosen with trail and error, I tried 1 and found the car was jerky.
    # I saw David use 0.02 and felt it was too week at times, so I selected 0.05. 
    # Went with 0.02 after more exirmenting
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

X_data = np.array(augmented_images)
y_data = np.array(augmented_measurements)

print('X_data shape:', X_data.shape)
print('y_data shape:', y_data.shape)

#Split Data into Traning and Validation 
# test_size = 0.2
# random_state = 0

# from sklearn.model_selection import train_test_split
# X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=test_size, random_state=random_state)
# print('X_train shape: ', X_train.shape)
# print('y_train shape: ', y_train.shape)
# print('X_valid shape: ', X_valid.shape)
# print('y_Valid Shape: ', y_valid.shape)

#setup model dependancies 
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
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(video_H, viedo_L, layers), output_shape=(video_H, viedo_L, layers)))
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
#model.fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
model.fit(X_data, y_data, validation_split=0.2, shuffle=True, nb_epoch=epoch)

model.save('nvidia03.h5')
exit()