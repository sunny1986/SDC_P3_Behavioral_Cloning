# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/nvidia_model.JPG "Model Visualization"
[image2]: ./examples/dist1.png "Data Distribution"
[image3]: ./examples/dist2.png "Data Distribution 1"
[image4]: ./examples/center_cam.JPG "Center Cam"
[image5]: ./examples/cropped_image.JPG "Cropped Image"
[image6]: ./examples/cropped_image1.JPG "Cropped Image 1"
[image7]: ./examples/left_cam.JPG "L cam Image"
[image8]: ./examples/center_cam1.JPG "C cam Image 1"
[image9]: ./examples/right_cam.JPG "R cam Image"
[image10]: ./examples/original.JPG "Original Image"
[image11]: ./examples/flipped.JPG "Flipped Image"
[image12]: ./examples/driving_gif.GIF "Driving Autonomously"


Results
---
![alt text][image12]


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

###### Files required to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

###### Driving autonomously
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

##### Model Architecture 

First a very basic neural network called "Trivial Model" consisting of a flattened image connected to a single ouput was added to verify if all the logistics of the project were working fine. The created model was used to run the simulator in autonomous mode and upto epoch 7 the validation loss was being reduced to approx. 2988.5543. The simulator driving was bad.

Next a lambda layer was also added to normalize the data using a Keras lambda layer (model.py line 101) which reduced the validation loss drastically to approx. 4.5865. The simulator driving was still bad, was turning hard right and away from the road on start

I also tried comma.ai model as suggested by one of mentors on the forum and was able to achieve validation loss of approx. 0.0185.

But reading more about it in forums and discussions suggested the Nvidia model was being used more. Started using that as the model.

My model consists of a modified version of Nvidia's CNN architecture used in their paper [End to End Learning for Self Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 103-107) subsampled by 2x2 & 1x1. 

* Attempts to reduce overfitting in the model

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model (model.py lines 103 onwards). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

* Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

* Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The importance of the training data was learnt throughout the training process and the details of which are described in the next section of Training Strategy.

![alt text][image1]

###### Nvidia's CNN architecture 


##### Training Strategy

* Initially the given training data was used to get a start and later on custom data was given to the model by running the simulator in training mode and collectiong images and corresponding steering angles.

* To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

* I order to get rid of unnecessary data, I added a layer in the model (model.py lines 102) to crop the images being fed to it as shown below.

![alt text][image5] ![alt text][image6]

* The driving improved a little bit and was able to drive straight until the first left turn but still the car was biased towards the right side.

* In order to break this bias and generalize better, I tried to augment the training dataset by adding L&R camera images and flipping the images using np.fliplr() function. Using L & R camera images was done in order to let the model learn that the car needs to turn right or left more in order to get to the middle of the road. Hence appropriate steering angle corrections were also made for L & R camera images as seen in lines 41 & 42 of model.py file

* Figures below show Left, Center & Right camera images

![alt text][image7] ![alt text][image8] ![alt text][image9]

* Figures below show original and flipped camera images

![alt text][image10] ![alt text][image11]

* When I added more data by adding L&R camera images and flipping, the CPU ran out of memory sending this to the network. Enter generators!

* Since it was my first time using generators took me some time to get the generator working correctly. Used augementation inside the generator but was getting same number of training images which should not be the case since I was augmenting using L&R camera images and flipping. My implementation was wrong here.

* But then I started to augment images outside the generator in a few lines of code. This stretegy worked out and I started feeding images and steering angle measurements to the generator instead of lines from csv file.

* As the size of the csv file grew the cv2.imread() function in the augmentation section threw 'Out of Memory' error. Suggested by someone on the forums, I used matplotlib.image.imread() function and it looked a lot better in handling memory utilization.

* Finally got the generator working with flipping augmentation and batches of data feeding the model. Understood the mechanisms of generator much better and importance of it since using a regular return statement would cause memory starvation pretty quickly.

* Much better driving, went past the bridge, past the sharp left turn, but went straight into the lake on right turn. Quality of training set here was 2 FWD laps, 2 REV laps no recovery data yet.

* A lot of times during driving in autonomous mode the car would drive straight and not turn and like above go straight into a wall or lake.

* Hence I started to look at the distribution of my training dataset and found that the network would be too biased to 0 & +- 0.2 rad of steering angle as shown below. Hence the code was modified (model.py line 65) in order to reduce any bias towards driving straight since the car would keep crashing at turns. I used only small amount data for driving straight (model.py line 77)

![alt text][image2]

* Reading about it more and discussing with other students on the forum I understood that supressing the 0 angle images or not using much was helpful which resulted in the distribution shown below. This really made a big difference in the performance and improved the training a lot more.

![alt text][image3]

* The importance of quality training was understood since it affected the driving a lot. Hence I focussed more on "tough areas" of the track like turns and different road quality like on the bridge which looks like a road made out of bricks compared to black tar road. 

* It was remarkable to see how the car recovered and got back onto the center of the road after training in those "tough areas". Earlier it would just crash. This really showed me how the network was "learning". Awesome!

* After the data collection process the numbers looks as below:

| Parameter										|     Count				| 
|:---------------------------------------------:|:---------------------:| 
| Total of lines of samples in csv file 		| 19678 samples			| 
| Adding L&R camera images 19678 X 3			| 59034 images 			|
| After flipping images 						| 118068 images			|

* With the help of generator I was able to randomly shuffle and send these images and their steering angles in batches of 1500 samples to train on for 3 epochs since that gave an optimum amount of training and validation loss and good driving.

* Finally after training the model in tricky areas of the track I was able to let the car drive autonomously the full track without leaving the road.
