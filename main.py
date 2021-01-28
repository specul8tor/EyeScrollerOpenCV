import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while 1:
	_, frame = cap.read()
	gray_picture = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = detector(gray_picture)
	for face in faces:  
		x,y = face.left(), face.top() 
		w,h=face.right()-x,face.bottom()-y
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		gray_face = gray_picture[y:y+h, x:x+w]
		face = frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(gray_face)
		for (ex,ey,ew,eh) in eyes: 
			cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)
	cv2.imshow("Frame",frame)
	if(cv2.waitKey(1)== ord('a')):
		print('pressed a')
		break   

cap.release()
cv2.destroyAllWindows()