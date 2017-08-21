import csv
import cv2
import numpy as np

lines = [] 
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line) 

images=[]
measurments = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data/IMG/' + filename
	image = cv2.imread(current_path) 
	image.append(image)
	measurement = float(line[3])
	measuremnets.append(measurement)

X_train = np.array(images) 
y_train = np.array(measurements)

from keras.models import Sequential
from keraslayers import Flatten, Dense 

model = Sequential()
model.add(Flatten(input_shap=(160,320,3)))
model.add(Dense(1))

