import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

clf = None
faces = {"Modi":0, "Kejriwal":1}

# faces = {"Kejriwal":1}

def train():
	x_train = []
	y_train = []
	for i in faces:
		# print i
		for j in os.listdir(i):
			img_path = i+"/"+j
			img_temp = cv2.imread(img_path)
			img = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

			face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
			# face_detect = cv2.CascadeClassifier('lbpcascade_profileface.xml')
			# face_detect = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
			# face = face_detect.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, flags= cv2.CASCADE_SCALE_IMAGE, minSize=(30,30))
			face = face_detect.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

			(x,y,a,b) = face[0]

			box = face[0]
			f_det = img[y:y+a, x:x+b]

			if f_det is not None:
				x_train.append(f_det)
				y_train.append(faces[i])

			# print len(face),img_path


	return x_train, y_train



def test_image(img_path):
    l = [0,0,0]
    img_temp = cv2.imread(img_path)
    img = img_temp.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # face_detect = cv2.CascadeClassifier('lbpcascade_profileface.xml')
    # face_detect = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    # face = face_detect.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, flags= cv2.CASCADE_SCALE_IMAGE, minSize=(30,30))
    face = face_detect.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    # print len(face)

    if len(face)==0:
		save_path = "static/"+str(len(os.listdir("./static"))+1)+".jpg"
		cv2.imwrite(save_path,img_temp)
		return save_path,l
    else:
		save_path = "static/"+str(len(os.listdir("./static"))+1)+".jpg"
		l[0]=1
		for i in range(len(face)):
			(x,y,a,b) = face[i]
			f_det = img[y:y+a, x:x+b]
			out, conf = clf.predict(f_det)
			if out==0:
				# print out, conf
				# if conf<=0.45:
				# 	cv2.rectangle(img_temp, (x, y), (x+a, y+b), (0, 255, 255), 2)
				# else:
				l[1] = 1
				cv2.rectangle(img_temp, (x, y), (x+a, y+b), (0, 255, 0), 2)
			else:
				# print out, conf
				# if conf<=0.45:
				# 	cv2.rectangle(img_temp, (x, y), (x+a, y+b), (0, 255, 255), 2)
				# else:
				l[2] = 1
				cv2.rectangle(img_temp, (x, y), (x+a, y+b), (0, 0, 255), 2)
		# print save_path
		cv2.imwrite(save_path,img_temp)
		return save_path,l


def out(img_path):
	global clf
	x,y = train()
	# print "training"
	clf = cv2.face.LBPHFaceRecognizer_create()
	# clf = cv2.face.EigenFaceRecognizer_create()
	# clf = cv2.face.FisherFaceRecognizer_create()
	clf.train(x, np.array(y))
	# print "trained"
	save_path, l = test_image(img_path)

	# save_path = "static/"+str(len(os.listdir("./static"))+1)+".jpg"
	# cv2.imwrite(save_path,img_out)
	# plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
	# plt.savefig(save_path)
	print save_path
	return save_path, l

# print out("1.jpg")
