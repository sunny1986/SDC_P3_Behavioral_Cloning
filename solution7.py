import csv
import os
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg

# Read data from csv file
samples = []
with open('../data1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# Augment data set by adding  L&R camera images and corrected steering angles		
correction = 0.2
aug_samples = []
for sample in samples:	
	for i in range(3):
		if i == 1:
			aug_samples.append([sample[i],float(sample[3]) + correction])
		elif i == 2:
			aug_samples.append([sample[i],float(sample[3]) - correction])
		else:
			aug_samples.append([sample[i],float(sample[3])])

# Verify augmentation
print(len(samples))
print(len(aug_samples))
			
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(aug_samples, test_size=0.2)
	
def generator(samples, batch_size):
	num_samples = len(samples)   
	batch_size = int(batch_size/2)
	while 1: # let the generator loop forever and never terminate
		shuffle(samples)
		for offset in range(0, num_samples):
			batch_samples = samples[offset:offset+batch_size]
					
			images = []
			measurements = []
			for batch_sample in batch_samples:        
				current_path = batch_sample[0]					
				image = mpimg.imread(current_path)					
				measurement = float(batch_sample[1])					
				images.append(image)						
				measurements.append(measurement)	

			img_batch = []
			angle_batch = []
			for img, angle in zip(images, measurements):                
				img_batch.append(img)
				angle_batch.append(angle)
				img_batch.append(np.fliplr(img))
				angle_batch.append(-angle)					

			#print("No.of flipped images=",len(img_batch))

			X_train = np.array(img_batch)
			y_train = np.array(angle_batch)						

			yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 36)
validation_generator = generator(validation_samples, batch_size=36)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers import Convolution2D, Dropout, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
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

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)),input_shape=(160,320,3)))
model.add(Conv2D(24, 5, 5, border_mode='valid', activation='tanh', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, border_mode='valid', activation='tanh', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, border_mode='valid', activation='tanh', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='tanh', subsample=(1, 1)))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='tanh', subsample=(1, 1)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1162, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))
#model.summary()

model.compile(loss='mae', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=3)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
	validation_data=validation_generator, 
	nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')