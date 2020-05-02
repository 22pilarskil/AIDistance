import numpy as np 
import cv2
import json
import keras
import time
import os
from mtcnn import MTCNN

mtcnn = MTCNN()

os.system("sh model_download.sh")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

import tensorflow as tf

def build_model(model_path, output_tensor_names, input_tensor_name):
	graph_def = None

	with tf.gfile.FastGFile(model_path, "rb") as graph_file:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(graph_file.read())

	sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
	tf.import_graph_def(graph_def, name='')

	output_tensors = [sess.graph.get_tensor_by_name(name) for name in output_tensor_names]
	input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

	return {"sess": sess, "output_tensors": output_tensors, "input_tensor": input_tensor}

def detect_faces(img):
	if isinstance(img, str):
		return mtcnn.detect_faces(cv2.imread("/Users/michaelpilarski/Desktop/1.jpg"))
	else:
		return mtcnn.detect_faces(img)

rcnn = build_model(
	model_path="frozen_inference_graph.pb",
	output_tensor_names=["detection_boxes:0", "detection_scores:0", "detection_classes:0"],
	input_tensor_name="image_tensor:0"
	)

facenet = build_model(
	model_path="20180402-114759.pb",
	output_tensor_names=["embeddings:0"],
	input_tensor_name="input:0"
	)


cap = cv2.VideoCapture(0)
while True:
	ret_val, img = cap.read()
	expanded = np.expand_dims(img, axis=0) #faster_rcnn requires this

	start = time.time()
	boxes, scores, classes = list(rcnn["sess"].run(rcnn["output_tensors"], feed_dict={rcnn["input_tensor"]: expanded}))
	print(time.time()-start)
	
	combined = zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes))
	#ensure box points provided, class is human, confidence greater than .5
	try:
		boxes, scores, classes = zip(*(list(filter(lambda x: sum(x[0]) > 0 and x[2] == 1 and x[1] > .5, combined)))) 
		bounds = []
		for box in boxes:
			bounds.append([int(box[i] * img.shape[i%2]) for i in range(len(box))])

		for bound in bounds:
			cv2.rectangle(img, (bounds[-1][1],bounds[-1][0]), (bounds[-1][3],bounds[-1][2]), color=(255,0,0), thickness=1)
		time.sleep(.1)
		cv2.imshow(str(len(boxes)), img)

	except ValueError: #nothing detected
		cv2.imshow("0", img)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

	print(len(boxes))

cap.release()
cv2.destroyAllWindows()





