import numpy as np                            #contains numerical function
import os                                            #direct files access
#import six.moves.urllib as urllib
import sys                                          #system file access
import tarfile                                       #to extract and compress file
import tensorflow as tf
import zipfile
import sys
import time                                           # to check time stamp
import cv2                                            # open cV computer vision
#import pygame
import requests                                    # to fetch data from internet

from collections import defaultdict
from io import StringIO                          
import matplotlib.pyplot as plt  
from PIL import Image                          # to work on image
from utils import label_map_util
from utils import visualization_utils as vis_util           # both are secondary package of tensor flow

#MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'                      #this is NAME OF pretrained MODEL to clasify, feature extraction

# high accuracy but very slow
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'

MODEL_FILE = MODEL_NAME + '.tar.gz'                                       #SAARE FILES TAR FILES K FORM M H ISISLIYE TAR.Z USE KIYA       
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'            #CKPT IS THE ACTUAL MODEL 

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')     

NUM_CLASSES = 90

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
 file_name = os.path.basename(file.name)                   #HR FILE KA ADDRESS CHECK KR RHE H
 if 'frozen_inference_graph.pb' in file_name:                
   tar_file.extract(file, os.getcwd())
   
detection_graph = tf.Graph()                               
with detection_graph.as_default():   
 od_graph_def = tf.GraphDef()
 with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
   serialized_graph = fid.read()
   od_graph_def.ParseFromString(serialized_graph)
   tf.import_graph_def(od_graph_def, name='')            
   
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
   od_graph_def = tf.GraphDef()
   with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
       serialized_graph = fid.read()
       od_graph_def.ParseFromString(serialized_graph)
       tf.import_graph_def(od_graph_def, name='')
sess = tf.Session(graph=detection_graph)

# Defining input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def region_of_interest(img, vertices):                           #to draw box
   mask = np.zeros_like(img)   
   if len(img.shape) > 2:
       channel_count = img.shape[2]
       ignore_mask_color = (255,) * channel_count
   else:
       ignore_mask_color = 255   
   cv2.fillPoly(mask, vertices, ignore_mask_color)
   masked_image = cv2.bitwise_and(img, mask)
   return masked_image

#camlink = "htttp://192.168.43.219:8080/shot.jpg"
#reqvid = requests.get(camlink)

video = cv2.VideoCapture(0)                                       #open cv function to capture video

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(video.get(3))                                      
frame_height = int(video.get(4))
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5 , (frame_width,frame_height))



try:
  
  while(video.isOpened()):

      ret, frame = video.read()
      stime = time.time()
      objects = []
      class_str = ""
      frame_width = frame.shape[0]
      frame_height = frame.shape[1]

      rows, cols = frame.shape[:2]

      left_boundary = [int(cols*0.40), int(rows*0.95)]
      left_boundary_top = [int(cols*0.40), int(rows*0.20)]
      right_boundary = [int(cols*0.60), int(rows*0.95)]
      right_boundary_top = [int(cols*0.60), int(rows*0.20)]

      bottom_left  = [int(cols*0.20), int(rows*0.95)]
      top_left     = [int(cols*0.20), int(rows*0.20)]
      bottom_right = [int(cols*0.80), int(rows*0.95)]
      top_right    = [int(cols*0.80), int(rows*0.20)]
      vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
      copied = np.copy(frame)
      interested=region_of_interest(copied,vertices)

      frame_expanded = np.expand_dims(interested, axis=0)

      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: frame_expanded})

      vis_util.visualize_boxes_and_labels_on_image_array(
          frame,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8,
          min_score_thresh=0.78)

      ymin = int((boxes[0][0][0]*frame_width))
      xmin = int((boxes[0][0][1]*frame_height))
      ymax = int((boxes[0][0][2]*frame_width))
      xmax = int((boxes[0][0][3]*frame_height))

      ymin_str='y min  = %.2f '%(ymin)
      ymax_str='y max  = %.2f '%(ymax)
      xmin_str='x min  = %.2f '%(xmin)
      xmax_str='x max  = %.2f '%(xmax)

      print(scores.max())
#     
#      if(xmin >= left_boundary[0]):
#          print("move LEFT - 1st !!!")
#          cv2.putText(frame,'Move LEFT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
          #txt = "Move left"
          #language = 'en'
          #obj = gTTS(text=txt, lang=language, slow=False)
          #obj.save("audio.mp3")
          #playsound.playsound('audio.mp3')
          #os.remove("audio.mp3")
          
          #time.sleep(1)

#      elif(xmax <= right_boundary[0]):
#          print("move Right - 2nd !!!")
#          cv2.putText(frame,'Move RIGHT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

#      elif(xmin <= left_boundary[0] and xmax >= right_boundary[0]):
#          print("STOPPPPPP !!!! - 3nd !!!")
#          cv2.putText(frame,' STOPPPPPP!!!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

#      cv2.line(frame,tuple(left_boundary),tuple(left_boundary_top), (255, 0, 0), 5)
#      cv2.line(frame,tuple(right_boundary),tuple(right_boundary_top), (255, 0, 0), 5)

      out.write(frame)

      cv2.imshow('object detection', frame)
      if cv2.waitKey(1) == ord('q'):
        cap.release() 
        cv2.destroyAllWindows()
        break
except:
  pass
