#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from flask import Flask, render_template, Response,jsonify
from imutils import face_utils
import cv2, imutils, dlib, numpy as np, os, tensorflow as tf, sys
import struct, collections
import projectlibrary as plib

app = Flask(__name__)

################# Import Initial Parameters #################
status_text='Welcome.. System Ready!!'
detector = dlib.get_frontal_face_detector()
pl=''
gls=[]
predictor = dlib.shape_predictor('models/face-detector.dat')
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])


@app.route('/')
def index():
	return render_template('home.html')

def image_resize(image):
	r = 100.0 / image.shape[1]
	dim = (100, int(image.shape[0] * r))
	# perform the actual resizing of the image and show it
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized
	
def get_frame():
	####################### Triggering the Camera #######################
	camera=cv2.VideoCapture(0)
	blink=0
	COUNTER = 0
	EYE_AR_THRESH = 0.22#0.2
	EYE_AR_CONSEC_FRAMES = 10
	classification_flag=0
	feedback_flag=0
	classify_count=0
	fr=1
	first_letter=[]
	first_probability=[]
	second_letter=[]
	second_probability=[]
	firstdecision='na'
	seconddecision='na'
	feedback_count=0
	feedback_decisions=[]
	predicted_letter=''
	
	while True:
		########################## Capturing Frame by Frame ##############################
		grabbed, frame = camera.read()
		global status_text
		status_text='Camera is running ..'
		size = frame.shape
		h=size[0]
		w=size[1]
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		focal_length = size[1]
		center = (size[1]/2, size[0]/2)
		camera_matrix = np.array(
								 [[focal_length, 0, center[0]],
								 [0, focal_length, center[1]],
								 [0, 0, 1]], dtype = "double"
								 )
		############################### Face Detection #################################
		if len(rects) != 0:
			status_text='Face Detected!'
			maxArea = 0
			maxRect = None
			for rect in rects:
				if rect.area() > maxArea:
					maxArea = rect.area()
					maxRect = [rect.left(),rect.top(),rect.right(),rect.bottom()]
			
			rect = dlib.rectangle(*maxRect)
			shape = predictor(gray, rect)
		
		################################ Blink Detection ##############################
			if blink==0:
				left_eye,right_eye,ear=plib.eyes_detection(shape)
				cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
				status_text='Eyes Detected!'
				if ear < EYE_AR_THRESH:
					COUNTER += 1
				else:
					if COUNTER >= EYE_AR_CONSEC_FRAMES:
						blink= 1
						status_text='Blink Detected!'
					COUNTER = 0
			elif blink==1:
				landmarks = plib.dlibLandmarksToPoints(shape)
				landmarks = np.array(landmarks)
				for (x, y) in landmarks:
					cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
				if ((classification_flag==0) or (feedback_flag==1)):
				########################### Hand Detection ##################################
					ix = landmarks[32][0]
					fx = landmarks[34][0]
					iy = landmarks[29][1]
					fy = landmarks[30][1]

					# Take a patch on the nose
					tempimg = frame[iy:fy,ix:fx,:]

					# Compute the mean image from the patch
					meanimg = np.uint8([[cv2.mean(tempimg)[:3]]])
					skinRegionycb,skinycb = plib.findSkinYCB(meanimg, frame)
					maskedskinycb = plib.applyMask(skinycb, landmarks)
					#cv2.putText(skinycb, "YCrCb", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
					##cv2.imshow('masked',maskedskinycb)
					##cv2.imshow("YCrCb",skinRegionycb)
					x1=int(landmarks[0][0])
					
					#cv2.circle(frame, (400,200),10, (255,0,0),-1)

					newframe=frame[:,0:x1]
					maskedframe=maskedskinycb[:,0:x1]
					handregion=skinRegionycb[:,0:x1]
					drawing,xh,yh,wh,hh=plib.handconvex(handregion,newframe,maxArea)
					if ((xh!=0) and (yh!=0) and (wh!=0) and (hh!=0)):
						status_text='Hand Detected..'
						crphand=maskedframe[yh:yh+hh,xh:xh+wh]
						r = 90.0 / crphand.shape[1]
						dim = (90, int(crphand.shape[0] * r))
						resized = crphand
						##cv2.imshow('hand',crphand)
					######################## Hand Classification ########################
						if classify_count<11:
							if (fr%2)==0:
								#print(classify_count)
								#imgencode=cv2.imencode('.jpg',resized)[1]
								#stringData=imgencode.tostring()
								resized = cv2.resize(resized, (224,224))
								np_image_data = np.asarray(resized)
								np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
								np_final = np.expand_dims(np_image_data,axis=0)
								
								predictions=plib.handclassify(np_final)
								letter,confidence=predictions[0]
								#print(letter)
								first_letter.append(letter)
								first_probability.append(confidence)
								letter,confidence=predictions[1]
								second_letter.append(letter)
								second_probability.append(confidence)
								#if confidance>0.70:
								#probabilities.append(confidance)
								classify_count=classify_count+1
						else:
							status_text='Hand Gestures Recognized..'
							classification_flag=1
							letters_counter=collections.Counter(first_letter)
							#print(letters_counter)
							letters=sorted(letters_counter.items(), key=lambda x: x[1])
							#print(letters)
							predicted_letter,freq=letters[len(letters)-1]
							global pl
							pl='Predicted letter : '+predicted_letter
							#print('Predicted letter: '+predicted_letter)
				################################ Head Pose Estimation ####################
				else:
					status='Feedback Mode..'
					up_extreme=(landmarks[19][0],landmarks[19][1]-10)
					left_extreme=landmarks[0]
					right_extreme=landmarks[16]
					image_points = np.array([
								landmarks[30],     # Nose tip
								landmarks[8],     # Chin
								landmarks[36],     # Left eye left corner
								landmarks[45],     # Right eye right corne
								landmarks[48],     # Left Mouth corner
								landmarks[54]      # Right mouth corner
							], dtype="double")
					#for point in image_points:
					#	x,y=point
					#	cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
					
					dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
					(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)
					##print(rotation_vector)
					(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
					for p in image_points:
						cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
				 
				 
					p1 = ( int(image_points[0][0]), int(image_points[0][1]))
					p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
					 
					cv2.line(frame, p1, p2, (255,0,0), 2)
					
					cv2.line(frame,(0,up_extreme[1]),(w,up_extreme[1]),(0,0,255),1)
					cv2.line(frame,(left_extreme[0],0),(left_extreme[0],h),(0,0,255),1)
					cv2.line(frame,(right_extreme[0],0),(right_extreme[0],h),(0,0,255),1)
					
					yes_left= left_extreme[0]
					yes_top=0
					yes_width=abs(left_extreme[0]-right_extreme[0])
					yes_height=up_extreme[1]
					
					yes_mid=(int((left_extreme[0]+right_extreme[0])/2),int(((up_extreme[1]+0)/2)))
					cv2.putText(frame, "YES", (yes_mid[0], yes_mid[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
					
					leftno_left= 0
					leftno_top=left_extreme[1]
					leftno_width=left_extreme[0]
					leftno_height=abs(left_extreme[1]-h)
					
					leftno_mid=(int((left_extreme[0]+0)/2),int(((up_extreme[1]+h)/2)))
					cv2.putText(frame, "NO", (leftno_mid[0], leftno_mid[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
					
					rightno_left= right_extreme[0]
					rightno_top=up_extreme[1]
					rightno_width=abs(w-right_extreme[0])
					rightno_height=abs(up_extreme[1]-h)
					
					rightno_mid=(int((right_extreme[0]+w)/2),int(((up_extreme[1]+h)/2)))
					cv2.putText(frame, "NO", (rightno_mid[0], rightno_mid[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
					#firstdecision='na'
					#seconddecision='na'
					if feedback_count<30:
						if ((yes_left<p2[0]) and (p2[0]<(yes_left+yes_width))):
							if ((yes_top<p2[1]) and (p2[1]<(yes_top+yes_height))):
								cv2.rectangle(frame,(left_extreme[0],up_extreme[1]),(right_extreme[0],0),(96,247,66),-1)
								cv2.putText(frame, "YES", (yes_mid[0], yes_mid[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
								feedback_decisions.append('yes')
								feedback_count=feedback_count+1	
						
						if ((leftno_left<p2[0]) and (p2[0]<(leftno_left+leftno_width))):
							if ((leftno_top<p2[1]) and (p2[1]<(leftno_top+yes_height))):
								cv2.rectangle(frame,(left_extreme[0],up_extreme[1]),(0,h),(57,27,247),-1)
								cv2.putText(frame, "NO", (leftno_mid[0], leftno_mid[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
								feedback_decisions.append('no')
								feedback_count=feedback_count+1
								
						
						if ((rightno_left<p2[0]) and (p2[0]<(rightno_left+rightno_width))):
							if ((rightno_top<p2[1]) and (p2[1]<(rightno_top+rightno_height))):
								cv2.rectangle(frame,(right_extreme[0],up_extreme[1]),(w,h),(57,27,247),-1)
								cv2.putText(frame, "NO", (rightno_mid[0], rightno_mid[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
								feedback_decisions.append('no')
								feedback_count=feedback_count+1
					else:
						counter=collections.Counter(feedback_decisions)
						feedback=sorted(counter.items(), key=lambda x: x[1])
						feedback,freq=feedback[len(feedback)-1]
						if feedback=='yes':
							if firstdecision=='na':
								firstdecision='yes'
							elif firstdecision=='no':
								seconddecision='yes'
						elif feedback=='no':
							if firstdecision=='na':
								firstdecision='no'
							elif firstdecision=='no':
								seconddecision='no'
						#print('firstdecision: '+firstdecision)
						#print('seconddecision: '+seconddecision)
						if firstdecision=='yes':
							#print('first decision is OK.') 
							#print('Next letter please..')
							gls.append(predicted_letter)
							pl='Recognition confirmed as '+predicted_letter
							pl=''
							firstdecision='na'
							classification_flag=0
							classify_count=0
							first_letter=[]
							second_letter=[]
						elif (firstdecision=='no') and (seconddecision=='na'):
							# scoring model
							predicted_letter,freq=letters[len(letters)-2]
							#print('Next Predicted letter: '+predicted_letter)
							pl='Next predicted letter:  '+predicted_letter
							first_letter=[]
							second_letter=[]
						elif (firstdecision=='no') and (seconddecision=='no'):
							#print('lets try once again')
							firstdecision='na'
							seconddecision='na'
							classification_flag=0
							classify_count=0
							pl=''
							first_letter=[]
							second_letter=[]
						elif (firstdecision=='no') and (seconddecision=='yes'):
							#print('I got it thanks')
							#print('Next letter please..')
							firstdecision='na'
							seconddecision='na'
							classification_flag=0
							classify_count=0
							pl='Recognition confirmed as '+predicted_letter
							gls.append(predicted_letter)
							first_letter=[]
							second_letter=[]
						feedback_decisions=[]
						feedback_count=0
		else:
			status_text='No Faces Detected..'
			
		fr=fr+1
		imgencode=cv2.imencode('.jpg',frame)[1]
		stringData=imgencode.tostring()
		##cv2.imshow('output',frame)
		#key = cv2.waitKey(1) & 0xFF
		#if key == ord("q"):
		#	break
		yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
	del(camera)

@app.route('/status')
def status():
	status_result = {'status': status_text, 'predicted_letter': pl}
	return jsonify(status_result)
@app.route('/letters')
def letters():
    if len(gls)>1:
        letters_str=''.join(gls)
        return jsonify(ltr=letters_str)
    elif len(gls)==1:
        letters_str=gls[0]
        return jsonify(ltr=letters_str)
    else:
        return jsonify(ltr='None')
@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, threaded=True)
