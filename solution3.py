import csv
import os
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

samples = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

print(len(samples))
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
correction = 0.2

def generator(samples, batch_size=33):
	#num_samples = len(samples)
	num_samples = 11
	while 1: # let the generator loop forever and never terminate
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			measurements = []
			for batch_sample in batch_samples:
				"""current_path = batch_sample[0]					
				image = cv2.imread(current_path)				
				measurement = float(line[3])	
				images.append(image)				
				measurements.append(measurement)"""
				for i in range(3):
					current_path = batch_sample[i]					
					image = cv2.imread(current_path)					
					if i == 1:
						measurement = float(line[3]) + correction
					elif i == 2:
						measurement = float(line[3]) - correction
					else:
						measurement = float(line[3])
					images.append(image)						
					measurements.append(measurement)	
				
			#print("No.of images : ",len(images))
			
			X_train = np.array(images)
			y_train = np.array(measurements)						
			
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=33)
validation_generator = generator(validation_samples, batch_size=33)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers import Convolution2D, Dropout, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2

"""
# Trivial Model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
"""
"""
# Comma.ai model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
"""
"""
modified NVIDIA model
"""

c2d2_p = {"border_mode": "valid",
		  "activation": "tanh",
		  "subsample": (2, 2)}
c2d1_p = {"border_mode": "valid",
		  "activation": "tanh",
		  "subsample": (1, 1)}
d_p = {"activation": "tanh", "W_regularizer": l2(0.001)}
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24, 5, 5, input_shape=(160,320,3), **c2d2_p))
model.add(Dropout(0.5))
model.add(Conv2D(36, 5, 5, **c2d2_p))
model.add(Dropout(0.5))
model.add(Conv2D(48, 5, 5, **c2d2_p))
model.add(Dropout(0.5))
model.add(Conv2D(64, 3, 3, **c2d1_p))
model.add(Dropout(0.5))
model.add(Conv2D(64, 3, 3, **c2d1_p))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50, **d_p))
model.add(Dropout(0.5))
model.add(Dense(20, **d_p))
model.add(Dropout(0.3))
model.add(Dense(10, **d_p))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=3)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
		validation_data=validation_generator, 
		nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')