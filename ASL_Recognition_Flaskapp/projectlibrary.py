from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.spatial import distance as dist
import imutils
import dlib
import cv2
import numpy as np
import argparse
from imutils import face_utils
import os
import tensorflow as tf, sys

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
leftEye = [36, 37, 38, 39, 40, 41]
rightEye = [42, 43, 44, 45, 46, 47]
mouth = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 ]
leftBrows = [17, 18, 19, 20, 21]
rightBrows = [22, 23, 24, 25, 26]

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
model_file = "models/retrained_graph_current_use.pb"
label_file = "models/retrained_labels_current_use.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"
graph = load_graph(model_file)
# kernal size for morphological opening 
k = 7
# convert Dlib shape detector object to list of tuples
def dlibLandmarksToPoints(shape):
	points = []
	for p in shape.parts():
		pt = (p.x, p.y)
		points.append(pt)
	return points
  
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def eyes_detection(shape):
	shape = face_utils.shape_to_np(shape)
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]
	leftEAR = eye_aspect_ratio(leftEye)
	rightEAR = eye_aspect_ratio(rightEye)
	ear = (leftEAR + rightEAR) / 2.0
	# compute the convex hull for the left and right eye, then
	# visualize each of the eyes
	leftEyeHull = cv2.convexHull(leftEye)
	rightEyeHull = cv2.convexHull(rightEye)
	return leftEyeHull,rightEyeHull,ear

def applyMask(skinImage, points):

  tempMask = np.ones((skinImage.shape[0], skinImage.shape[1]), dtype = np.uint8)
  
  temp = []
  for p in leftEye:
    temp.append(( points[p][0], points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in rightEye:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in leftBrows:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in rightBrows:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in mouth:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  return cv2.bitwise_and(skinImage, skinImage, mask = tempMask)

def findSkinYCB(meanimg, frame):

  # Specify the offset around the mean value
  CrOffset = 12#15 # My edit 12
  CbOffset = 12#15 # My edit 12
  YValOffset = 90#100 # My edit 90
  
  # Convert to the YCrCb color space
  ycb = cv2.cvtColor(meanimg,cv2.COLOR_BGR2YCrCb)[0][0]
  frameYCB = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)

  # Find the range of pixel values to be taken as skin region
  minYCB = np.array([ycb[0] - YValOffset,ycb[1] - CrOffset, ycb[2] - CbOffset])
  maxYCB = np.array([ycb[0] + YValOffset,ycb[1] + CrOffset, ycb[2] + CbOffset])

  # Apply the range function to find the pixel values in the specific range
  skinRegionycb = cv2.inRange(frameYCB,minYCB,maxYCB)

  # Apply Gaussian blur to remove noise
  skinRegionycb = cv2.GaussianBlur(skinRegionycb, (7, 7), 0)

  # Get the kernel for performing morphological opening operation
  
  #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
  #skinRegionycb = cv2.morphologyEx(skinRegionycb, cv2.MORPH_OPEN, kernel, iterations = 3)
  #skinRegionycb = cv2.dilate(skinRegionycb, kernel, iterations=3)
  
  
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)) #12,12
  skinRegionycb = cv2.morphologyEx(skinRegionycb, cv2.MORPH_CLOSE, kernel, iterations=3)

  # Apply the mask to the image
  skinycb = cv2.bitwise_and(frame, frame, mask = skinRegionycb)
  return skinRegionycb,skinycb

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
def handconvex(bin_img,img,faceArea):
	_,contours, hierarchy = cv2.findContours(bin_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	drawing = np.zeros(img.shape,np.uint8)
	max_area=0
	x=0
	y=0
	w=0
	h=0
	if contours:
		cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]
		#cnt=contours[ci]
		xi,yi,wi,hi = cv2.boundingRect(cnt)
		handArea=wi*hi
		if (handArea>(faceArea/2)):
			x,y,w,h=xi,yi,wi,hi
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			hull = cv2.convexHull(cnt)
			moments = cv2.moments(cnt)
			if moments['m00']!=0:
				cx = int(moments['m10']/moments['m00']) # cx = M10/M00
				cy = int(moments['m01']/moments['m00']) # cy = M01/M00
				centr=(cx,cy)       
				cv2.circle(img,centr,5,[0,0,255],2)       
			cv2.drawContours(drawing,[cnt],0,(0,255,0),2) 
			cv2.drawContours(drawing,[hull],0,(0,0,255),2) 

			cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
			hull = cv2.convexHull(cnt,returnPoints = False)
	return drawing,x,y,w,h

def handclassify(image_data):
	t=image_data
	#t = read_tensor_from_opencv(image_data,
	#							  input_height=input_height,
	#							  input_width=input_width,
	#							  input_mean=input_mean,
	#							  input_std=input_std)
	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name);
	output_operation = graph.get_operation_by_name(output_name);

	with tf.Session(graph=graph) as sess:
		results = sess.run(output_operation.outputs[0],
						{input_operation.outputs[0]: t})
	results = np.squeeze(results)
	top_k = results.argsort()[-5:][::-1]
	labels = load_labels(label_file)
	predictions=[]
	for i in top_k:
		predictions.append((labels[i], results[i]))
		print(labels[i], results[i])
	return predictions


def image_resize(image):
	r = 700.0 / image.shape[1]
	dim = (700, int(image.shape[0] * r))
	# perform the actual resizing of the image and show it
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
	input_name = "file_reader"
	output_name = "normalized"
	file_reader = tf.read_file(file_name, input_name)
	if file_name.endswith(".png"):
		image_reader = tf.image.decode_png(file_reader, channels = 3,
										name='png_reader')
	elif file_name.endswith(".gif"):
		image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
													name='gif_reader'))
	elif file_name.endswith(".bmp"):
		image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
	else:
		image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
											name='jpeg_reader')
	print(image_reader)
	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0);
	resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.Session()
	result = sess.run(normalized)
	return result

def read_tensor_from_opencv(image_data, input_height=299, input_width=299,
				input_mean=0, input_std=255):
	output_name = "normalized"
	image_reader = tf.image.decode_image(image_data, channels = None,name=None)
	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0);
	resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.Session()
	result = sess.run(normalized)
	return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def distance(x,y):
    import math
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)