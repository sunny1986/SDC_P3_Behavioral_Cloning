import csv
import os
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg

# Read data from csv file
samples = []
with open('../data2/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

print("csv samples",len(samples))

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

print("lcr samples",len(aug_samples))

# Extract images and measurements into lists
aug_images = []
aug_measurements = []
for aug_sample in aug_samples:
	aug_images.append(aug_sample[0])
	aug_measurements.append(aug_sample[1])
	
print("samples2images",len(aug_images))
print()

# Augment more by adding flipped images

def generator(aug_images, aug_measurements,sizeofbatch = 100):
	numofsamples = len(aug_images)	
	while 1:
		shuffle(aug_images,aug_measurements)
		for offset in range(0, numofsamples, sizeofbatch):
			imageset = aug_images[offset:offset+sizeofbatch]
			angleset = aug_measurements[offset:offset+sizeofbatch]
			
			img_batch = []
			angle_batch = []			
			for name,angle in zip(imageset,angleset):				
				img = mpimg.imread(name)
				img_batch.append(img)
				angle_batch.append(angle)
				img_batch.append(np.fliplr(img))
				angle_batch.append(-angle)			
			
			img_batch = np.array(img_batch)
			angle_batch = np.array(angle_batch)
			
			yield img_batch, angle_batch

train_sets = generator(aug_images,aug_measurements,sizeofbatch = 100)
#print(train_sets)
			
from sklearn.model_selection import train_test_split

train_samples, validation_samples, train_angles, valid_angles = train_test_split(train_sets, test_size=0.1)
	
def generator1(gen_images,gen_measurements,batch_size):
	num_samples = len(gen_images)
	while 1: # let the generator loop forever and never terminate
		shuffle(gen_images,gen_measurements)
		for offset in range(0, num_samples, batch_size):
			batch_images = gen_images[offset:offset+batch_size]
			batch_measurements = gen_measurements[offset:offset+batch_size]

			images = []
			measurements = []
			for batch_image,batch_measurement in zip(batch_images,batch_measurements):					
					images.append(batch_image)						
					measurements.append(batch_measurement)	
						
			X_train = np.array(images)
			y_train = np.array(measurements)				
			
			yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator1(train_samples,train_angles, batch_size=32)
validation_generator = generator1(validation_samples,valid_angles, batch_size=32)

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
# Nvidia Model
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
	nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.h5')