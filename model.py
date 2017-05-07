import csv
import os
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers import Convolution2D, Dropout, Cropping2D
from keras.layers.advanced_activations import ELU
import matplotlib.image as mpimg

# Extract image paths and steering angle data from csvfile
samples = []
samples = []
with open('../data1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
print()
print("Total samples from csv file=",len(samples))
print()

# Define steering angle correction to be used for L&R camera images
correction = 0.2
# Define arrays to be used later
c_paths = []
l_paths = []
r_paths = []
c_angles = []
l_angles = []
r_angles = []

# Separate extracted paths into center c, left l and right r paths
for sample in samples:
	c_paths.append(sample[0])
	l_paths.append(sample[1])
	r_paths.append(sample[2])
	
	c_angles.append(float(sample[3]))
	l_angles.append(float(sample[3]) + correction)
	r_angles.append(float(sample[3]) - correction)

# Function to augment images passed to it by flipping and returning 
# arrays of augmented and original images with respective augmented steering angles
def augment(images, angles):    
    images = np.array(images)
    angles = np.array(angles)
    
    aug_images = np.concatenate((images, np.fliplr(images)), axis=0)
    #Multiply all elements of angles array with -1
    flip_angles = [x * -1 for x in angles] 
    aug_angles = np.concatenate((angles,flip_angles),axis=0)
   
    return aug_images, aug_angles

# Define generator to process the whole training set
# and send images in batches to the model
def gen(batch_number,batch_size):

	images = []
	angles = []
	zero_counter = 0
	for i in range(batch_number*batch_size,(batch_number+1)*batch_size):
		if c_angles[i] != 0:
			images.append(mpimg.imread(c_paths[i]))
			images.append(mpimg.imread(l_paths[i]))
			images.append(mpimg.imread(r_paths[i]))

			angles.append(c_angles[i])
			angles.append(l_angles[i])
			angles.append(r_angles[i])
		else:
			zero_counter = zero_counter + 1
		# include every 100th 0 rad steering angle sample instead of all 
		# to break any bias towards 0 rad angle
		if zero_counter > 100:						
			images.append(mpimg.imread(c_paths[i]))
			images.append(mpimg.imread(l_paths[i]))
			images.append(mpimg.imread(r_paths[i]))

			angles.append(c_angles[i])
			angles.append(l_angles[i])
			angles.append(r_angles[i])
			zero_counter = 0
	# Augment data to let model generalize better
	X_train, y_train = augment(images,angles)
		
	yield shuffle(X_train,y_train)

"""
# Trivial Model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
"""

# Model definition - modified Nvidia's model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)),input_shape=(160,320,3)))
model.add(Conv2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
#model.summary()

# Batch size = no. of samples passed on to the generator from the training set
batch_size = 1500
batches = int(len(samples)/batch_size)
print()
print("Total batches=",batches)
print()
for batch_number in range(batches):
	print("batch_number=",batch_number)
	print()
	# Grab training data from generator
	training_batch = gen(batch_number,batch_size)
	X_train_y_train = (next(training_batch))
	X_train = X_train_y_train[0]
	y_train = X_train_y_train[1]
	model.fit(X_train, y_train, validation_split=0.1,shuffle=True, nb_epoch=3)
# Save model
model.save('model.h5')