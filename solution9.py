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
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

samples = []
samples = []
with open('../data1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
print()
print("Total samples from csv file=",len(samples))
print()
correction = 0.2
c_paths = []
l_paths = []
r_paths = []
c_angles = []
l_angles = []
r_angles = []
for sample in samples:
	c_paths.append(sample[0])
	l_paths.append(sample[1])
	r_paths.append(sample[2])
	
	c_angles.append(float(sample[3]))
	l_angles.append(float(sample[3]) + correction)
	r_angles.append(float(sample[3]) - correction)


def augment(images, angles):    
    images = np.array(images)
    angles = np.array(angles)
    
    aug_images = np.concatenate((images, np.fliplr(images)), axis=0)
    #Multiply all elements of angles array with -1
    flip_angles = [x * -1 for x in angles] 
    aug_angles = np.concatenate((angles,flip_angles),axis=0)
   
    return aug_images, aug_angles
    
def gen(batch_number,batch_size):    
    images = []
    angles = []
    for i in range(batch_number*batch_size,(batch_number+1)*batch_size):
        images.append(mpimg.imread(c_paths[i]))
        images.append(mpimg.imread(l_paths[i]))
        images.append(mpimg.imread(r_paths[i]))
                
        angles.append(c_angles[i])
        angles.append(l_angles[i])
        angles.append(r_angles[i])
    
    X_train, y_train = augment(images,angles)        
    yield shuffle(X_train,y_train)

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
model.add(Dense(100, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#model.summary()

batch_size = 1500
batches = int(len(samples)/batch_size)
print()
print("Total batches=",batches)
print()
for batch_number in range(batches):
	print("batch_number=",batch_number)
	print()
	training_batch = gen(batch_number,batch_size)
	X_train_y_train = (next(training_batch))
	X_train = X_train_y_train[0]
	y_train = X_train_y_train[1]
	model.fit(X_train, y_train, validation_split=0.1,shuffle=True, nb_epoch=3)
model.save('model.h5')