from imutils.video import VideoStream
import numpy as np
import sys
import argparse
import imutils
import time
import cv2
import mediapipe as mp

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

print("[INFO] starting video stream...")


vs = cv2.VideoCapture(0)
save_name = "output.mp4"
fps = 30
width = 1280
height = 720
output_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


time.sleep(2.0)

detected_objects = []
# loop over the frames from the video stream

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

	while vs.isOpened():

		ret, frame = vs.read()
		frame = imutils.resize(frame, width=800)
		
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		net.setInput(blob)
		detections = net.forward()

		blank_image = np.zeros((h, w,3), np.uint8)
		blank_image[:] = (0,0,0)      # (B, G, R)
		for i in np.arange(0, detections.shape[2]):
			try:
				confidence = detections[0, 0, i, 2]
				if confidence > 0.6:
					idx = int(detections[0, 0, i, 1])
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					label = "{}: {:.2f}%".format(CLASSES[idx],
						confidence * 100)

					if CLASSES[idx] == "person":
						detected_objects.append(label)
						cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
						cropped_human = frame[startY:endY, startX:endX]
						if len(cropped_human) > 0:
							bbbb = np.zeros((cropped_human.shape[0],cropped_human.shape[1],3), np.uint8)
							bbbb[:] = (0,0,0)      # (B, G, R)
							image = cv2.cvtColor(cropped_human, cv2.COLOR_BGR2RGB)
							image.flags.writeable = False
				
							results = pose.process(image)
				
							image.flags.writeable = True
							image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
				
							mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            	    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2), 
                                	mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                 	)

							mp_drawing.draw_landmarks(bbbb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                 )      
							frame[startY:endY, startX:endX] = image






						y = startY - 15 if startY - 15 > 15 else startY + 15
						cv2.putText(frame, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			except BaseException:
					pass
			
	
	# show the output frame
			cv2.imshow("Frame", frame)



		key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break


vs.release()    

cv2.destroyAllWindows()
