import cv2
import json
import keras
import time
import os
import gc
import argparse
import requests

import tensorflow as tf
import numpy as np

from mtcnn import MTCNN
from sklearn.neighbors import NearestNeighbors

mtcnn = MTCNN()

os.system("sh model_download.sh")


def build_model(model_path, output_tensor_names, input_tensor_name, placeholder=None):
    graph_def = None

    with tf.gfile.FastGFile(model_path, "rb") as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf.import_graph_def(graph_def, name='')

    output_tensors = [sess.graph.get_tensor_by_name(name) for name in output_tensor_names]
    input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

    if placeholder:
        placeholder = tf.placeholder(tf.float64,placeholder)

    return {"sess": sess, "output_tensors": output_tensors, "input_tensor": input_tensor, "placeholder":placeholder}

def detect_faces(img):
    if isinstance(img, str):
        return mtcnn.detect_faces(cv2.imread("/Users/michaelpilarski/Desktop/1.jpg"))
    else:
        return mtcnn.detect_faces(img)

def calc_overlap(bounds):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(bounds)
    distances, indices = nbrs.kneighbors(bounds)
    mask = np.ma.masked_equal(distances, 0.0, copy=False)
    min_dist = int(np.argmin(mask) / 2)

    indices_ = indices[min_dist]

    rect1 = bounds[indices_[0]]
    rect1_area = abs((rect1[2] - rect1[0]) * (rect1[3] - rect1[1]))

    rect2 = bounds[indices_[1]]
    rect2_area = abs((rect2[2] - rect2[0]) * (rect2[3] - rect2[1]))

    overlap_area = abs((max(rect1[1], rect2[1]) - min(rect1[3], rect2[3])) *
        (max(rect1[2], rect2[2]) - max(rect1[0], rect2[0])))
    overlap_percentage = overlap_area / min(rect1_area, rect2_area)

    if not len(bounds) == 1 and (overlap_percentage > .5 or max(rect1_area, rect2_area) / min(rect1_area, rect2_area) > 1.3):
        del(bounds[0 if rect1_area < rect2_area else 1])
        calc_overlap(bounds)
    else: 
        return bounds

def calc_centerpoints(bounds):
	centerpoints = 0
	for bound in bounds:
		centerpoints += (bound[3] + bound[1]) / 2
	return centerpoints / len(bounds)


def gstreamer_pipeline(capture_width=1280,capture_height=720,display_width=1280,display_height=720,framerate=60,flip_method=0):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink "
        "max-buffers=1 drop=True "
        % (capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--orientation", help="orientation of camera relative to entrance/exit",  type=str, default="right")
	parser.add_argument("--device", help="video or device to feed to inception_v2_coco", default=0)
	parser.add_argument("--location", help="location to post data to", default="Krogers")

	args = parser.parse_args()
	if bool(args.device):
		assert os.path.exists(args.device), "Please provide a valid path"

	heading = {
			"right": 
			{
				"centerpoint":lambda x: x > 0, 
				"heading":lambda x: x[0] > x[1]
			}, 
			"left": 
			{
				"centerpoint":lambda x: x < 0, 
				"heading":lambda x: x[0] < x[1]
			}
		}

	rcnn = build_model(
	    model_path="frozen_inference_graph.pb",
	    output_tensor_names=["detection_boxes:0", "detection_scores:0", "detection_classes:0"],
	    input_tensor_name="image_tensor:0",
	    placeholder=(1, None, None, 3)
	    )

	facenet = build_model(
	    model_path="20180402-114759.pb",
	    output_tensor_names=["embeddings:0"],
	    input_tensor_name="input:0"
	    )


	cap = cv2.VideoCapture(args.device)
	if not cap.isOpened():
	    cap = cv2.VideoCapture(gstreamer_pipeline(display_width=640, display_height=360, flip_method=0), cv2.CAP_GSTREAMER)

	frame_center = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
	last_centerpoint = np.inf if args.orientation == "right" else -np.inf

	current_val = 0
	people = 0
	missed_frames = 0
	entering = None

	while True:
	    ret_val, img = cap.read()
	    expanded = np.expand_dims(img, axis=0) #faster_rcnn requires this

	    boxes, scores, classes = list(rcnn["sess"].run(rcnn["output_tensors"], feed_dict={rcnn["input_tensor"]: expanded}))
	    
	    combined = zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes))
	    #ensure box points provided, class is human, confidence greater than .5

	    bounds = []
	    try:
	        boxes, scores, classes = zip(*(list(filter(lambda x: sum(x[0]) > 0 and x[2] == 1 and x[1] > .5, combined)))) 
	        for box in boxes:
	            bounds.append([int(box[i] * img.shape[i%2]) for i in range(len(box))])

	        bounds = calc_overlap(bounds) if len(bounds) > 1 else bounds

	        if (current_val < len(bounds)) and last_centerpoint:
	            current_centerpoint = calc_centerpoints(bounds)
	            entering = heading[args.orientation]["centerpoint"](current_centerpoint - frame_center) and \
	            	heading[args.orientation]["heading"]((last_centerpoint, current_centerpoint))
	            difference = len(bounds)-current_val
	            if entering:
		            people += difference
		            print("{} person(s) entered the room: {} persons in total".format(difference, people))
	            else:
	            	people -= difference
	            	r = requests.get("http://67.205.155.37:8000/setNumberofPeople", params={"people":people, "location":args.location})
	            	print("{} person(s) left the room: {} persons in total".format(difference, people))

	            current_val = len(bounds)

	        else:
	        	last_centerpoint = calc_centerpoints(bounds)


	        for bound in bounds:
	            color=(0,255,0) if entering else (0,0,255)
	            cv2.rectangle(img, (bound[1],bound[0]), (bound[3],bound[2]), color=color, thickness=4)
	            cv2.putText(img, "Entering" if entering else "Leaving", org=(bound[1], bound[3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
	            	fontScale=4, thickness=1, color=color)
	        missed_frames = 0

	    except (ValueError, TypeError) as e:
	    	missed_frames += 1 #nothing detected
	    	if missed_frames >= 6: #take into account error from inception_v2_coco
	    		current_val = 0
	    		last_centerpoint = 0 
	    	if isinstance(e, TypeError):
	    		break

	    if bounds:
	        cv2.imshow("Image", img)
	    else:
	        cv2.imshow("Image", img)


	    if cv2.waitKey(1) & 0xFF == ord("q"):
	        break

	cap.release()
	cv2.destroyAllWindows()