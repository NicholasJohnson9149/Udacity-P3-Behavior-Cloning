#**Behavioral Cloning** 

##Writeup July 2017 By Nicholas Johnson 

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write-up-img/Screenshot from 2017-08-28 13-58-04.jpg "Model Visualization"
[image2]: ./write-up-img/center_2017_08_20_23_52_40_375.jpg "Center Img"
[image3]: ./write-up-img/recovery-img-01.jpg "Recovery Image left"
[image4]: ./write-up-img/recovery-img-02.jpg "Recovery Image right"
[image5]: ./write-up-img/recovery-img-03.png "Recovery Image off track"
[image6]: ./write-up-img/center_2017_08_20_23_52_40_375.jpg "Normal Image"
#[image7]: ./write-up-img/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* Output Viedo of It Making Around the track at 9mph <a href="https://www.youtube.com/watch?v=jLts789TEbk">YouTube Link</a>

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on Nvidia's Drive one paper - Here is the output from my Nvida.py Model used to train the car. 

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 33, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

The model includes RELU layers to introduce nonlinearity nvidia.py, and the data is normalized in the model using a Keras lambda layer. 

What is most interesting about this model is that it's relatively simple yet extremely powerful for learning new things. It's also similar in many ways to the traffic sign recognition model LeNet. 

I first tried the LeNet model while going through the lecture videos but one David started talking about thsi paper I got excited. I read this <a href="http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf"> paper</a> a few months back deforming deciding I wanted to learn more by taking this course. watching this model take real driving input and then attempt to drive has been amazing. 
on line degrees in computer science


####2. Attempts to reduce over fitting in the model

The model contains dropout layers in order to reduce overfitting nvidia.py line ~81. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually, it worked really well so I believe the adam optimizers are more turned than my instances as of today. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road along with a lap going the opposite way. I found that using the larger data set often lead to what felt like over fitting. based on the results It's clear that more data and more work is needed to get a robust model capable of driving smoother. 


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the teh lectures, I thought this model might be appropriate because we had used it in teh past and it does very well with image and video data. The hidden layers allow for flexibility and each kernal gave opertuntes for refinement. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from over and under steering situations. These images show what a recovery looks like starting from the sides:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points. I collected 6 different data sets to try out different approaches to what data was good. Some data worked better than others in training the car to drive. I found that about 5-6 laps of data was great at training the model. This helped create more good laps than bad laps, and also allowed me to dedicate an entire lap to recovery data. I would pull the car off the the side of the road a bit and then start recoding it coming back to the center. I would then stop recording and then repeat the process for an entire lap. 

To augment the data set, data set 6 preformed the best for training, I also flipped images and angles thinking that this would help remove the left hand turn bias, I added some randomness to the flipped images as to not create a perfectly opposing data set, here is an image that has then been flipped:

![alt text][image6]
#![alt text][image7]

Etc ....

After the collection process, I had 14,116 data points. I then preprocessed this data by using a generator function break it smaller data sets as shown in the lectures for this assignment. 

I finally randomly shuffled the data set and put 20% of the data into a validation set and the remainder in the training set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by a graphing the epochs losses as they stabilized to a flat line. I used an adam optimizer so that manually training the learning rate wasn't necessary, this method work well enough to train the car. To grasp the learning rates effect more I should play around with a learning rate. 

There are a number of improvements that could be made to get the car to drive better on a sider range of environments, therefore I know some over fitting is accruing. There are a number of perimeters I could play with to achieve this and when I find more time plan to dial the model into something more robust. 
