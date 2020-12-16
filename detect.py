from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import dlib
import cv2
from util import euclidean_distance
from util import eye_aspect_ratio
from util import sound_alarm

shape_predictpr = "./shape_predictor_68_face_landmarks.dat"
alarm_path = "./alarm.wav"

EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictpr)

(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

video = VideoStream(0).start()

while True:
	frame = video.read()
	frame = imutils.resize(frame, width = 900)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detect = detector(gray, 0)

	for rect in detect:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

        # detect and secure left and right eyes
		left_eye = shape[left_eye_start:left_eye_end]
		right_eye = shape[right_eye_start:right_eye_end]

        # calculate EAR for both the eyes
		left_EAR = eye_aspect_ratio(left_eye)
		right_EAR = eye_aspect_ratio(right_eye)

        # average the EAR of both the eyes
		avg_EAR = (left_EAR + right_EAR) / 2.0

        # determine convex hull around the eyes
		left_eye_hull = cv2.convexHull(left_eye)
		right_eye_hull = cv2.convexHull(right_eye)

        # draw contours around both the eyes
		cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

		if avg_EAR < EAR_THRESHOLD:
			COUNTER += 1
			if COUNTER >= EAR_CONSEC_FRAMES:
				if not ALARM:
					ALARM = True
					if alarm_path != "":
						thread = Thread(target = sound_alarm(alarm_path), args = alarm_path)
						thread.deamon = True
						thread.start()
				cv2.putText(frame, "Drowsiness Detected! Wake up!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			COUNTER = 0
			ALARM = False
		cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(avg_EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("Pay Attention while Driving", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == 27:
		break

cv2.destroyAllWindows()
video.stop()