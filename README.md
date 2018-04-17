# Finger-Spelling-American-Sign-Language-Recognition-using-CNN
This project deals with recognition of finger spelling American sign language hand gestures using Computer Vision and Deep Learning. The system is hosted as web application using flask and runs on the browser interface.

## Getting Started
This application is built using Python programming language and runs on both Windows/ Linux platforms. Follow the instructions to run the application sucessfully.

### Prerequisites
Below Python packages are requiered for running this project

```
Python>3.4
OpenCV
Numpy
Flask
imutils
Dlib
Tensorflow
```
### Installing
To install all the required packages run the python script setup.py
```
python setup.py --device windows|linux|raspberrypi
```

For Linux systems, Opencv installation maynot be sucessfull using pip. Alternatively use "sudo apt-get install python-opencv" or install Opencv from sourse.
Here are some useful resources for installing opencv from source:
* [Ubuntu] https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
* [Raspberry Pi] https://www.pyimagesearch.com/2017/10/09/optimizing-opencv-on-the-raspberry-pi/

```
python run.py 
```
### Running the Application

After sucessful installation of prerequisites, execute the run.py python script to start the flask web service. 
In the browser (preffered Google Chrome) use http://localhost:5000 or http://0.0.0.0:5000 to interact with the application


## Usage
After sucessfully launching the application on your browser you can see the below interface to communicate with the gesture recognition system.

### System Flow
![System Flow](https://github.com/lokeshkumarVis/Finger-Spelling-American-Sign-Language-Recognition-using-CNN/blob/master/images/system_design.jpg)

#### Input
![Input](https://github.com/lokeshkumarVis/Finger-Spelling-American-Sign-Language-Recognition-using-CNN/blob/master/images/screenshots/input.png)

### Face Detection
![face detection](https://github.com/lokeshkumarVis/Finger-Spelling-American-Sign-Language-Recognition-using-CNN/blob/master/images/screenshots/face_detection.png)

### Trigger Gesture Recognition using Blink
![blink detection](https://github.com/lokeshkumarVis/Finger-Spelling-American-Sign-Language-Recognition-using-CNN/blob/master/images/screenshots/blink.png)

### Hand Detection
![hand detection](https://github.com/lokeshkumarVis/Finger-Spelling-American-Sign-Language-Recognition-using-CNN/blob/master/images/screenshots/hand_detection.png)

### Gesture Recognition
![Gesture Recognition](https://github.com/lokeshkumarVis/Finger-Spelling-American-Sign-Language-Recognition-using-CNN/blob/master/images/screenshots/gesture_recognition.png)

### Validate recognition using head pose
![validation yes](https://github.com/lokeshkumarVis/Finger-Spelling-American-Sign-Language-Recognition-using-CNN/blob/master/images/screenshots/validation_yes.png)


![validation no](https://github.com/lokeshkumarVis/Finger-Spelling-American-Sign-Language-Recognition-using-CNN/blob/master/images/screenshots/validation_no.png)

## Debugging Tips:
If you find any difficulty in running the application after sucessful installation of dependencies, Follow these debugging steps:
* Check if you have Tensorflow version 1.0 or later
* Check if port 5000 is allocated to any other service
* Clear the browsing chache and start a new session
* Force stop the flask application by pressing ctrl+c and re-launch the application

## Happy Learning !!
