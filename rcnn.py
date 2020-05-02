import numpy as np 
import cv2
import json
import tensorflow as tf
import keras
import time

model_path = "/Users/michaelpilarski/Desktop/frozen_inference_graph.pb"
graph_def = None

output_tensor_names = ["detection_boxes:0", "detection_scores:0", "detection_classes:0"]
input_tensor_name = "image_tensor:0"

with tf.gfile.FastGFile(model_path, "rb") as graph_file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph_file.read())

sess = keras.backend.get_session()
tf.import_graph_def(graph_def, name='')

output_tensors = [sess.graph.get_tensor_by_name(name) for name in output_tensor_names]
input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

#cap = cv2.VideoCapture(0)

img = cv2.imread("/Users/michaelpilarski/Desktop/1.jpg")
expanded = np.expand_dims(img, axis=0) #faster_rcnn requires this

start = time.time()
boxes, scores, classes = list(sess.run(output_tensors, feed_dict={input_tensor: expanded}))
print(time.time()-start)
combined = zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes))
#ensure box points provided, class is human, confidence greater than .5
boxes, scores, classes = zip(*(list(filter(lambda x: sum(x[0]) > 0 and x[2] == 1 and x[1] > .5, combined)))) 

boxes_list = []
for box in boxes:
	boxes_list.append([int(box[i] * img.shape[i%2]) for i in range(len(box))])
    
print(boxes_list)
for box in boxes_list:
	cv2.rectangle(img, (box[1],box[0]), (box[3],box[2]), color=(255,0,0), thickness=1)





