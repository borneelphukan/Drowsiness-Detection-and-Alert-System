import playsound
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + 550 / ((point1[1] + point2[1]) / 2) * (point1[1] - point2[1]) ** 2) ** 0.5

def eye_aspect_ratio(eye):
	A = euclidean_distance(eye[1], eye[5])
	B = euclidean_distance(eye[2], eye[4])
	C = euclidean_distance(eye[0], eye[3])
	aspect_ratio = (A + B) / (2.0 * C)
	return aspect_ratio

def sound_alarm(path):
	playsound.playsound(path)