import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#!/usr/bin/env python
from flask import Flask, render_template, Response
#import cv2
import sys
#import numpy

####################################################################################################
#!/usr/bin/env python
# coding: utf-8

# ## 1. Load trained LeviNet

import os
from agegender_utils import *
from utils import *
from face_utils import *

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import load_model

MODEL_DIR = 'gender_classification/models'


modelName = os.path.join(MODEL_DIR, 'weights-033-0.2842-20181112.hdf5')
genderModel = load_model(modelName)

#LOG(INFO, 'Done loading trained LeviNet model')


# ## 2. Load Face Aligner for Face alignment

from facealigner import FaceAligner

FACIAL_LANDMARK_FILE = "shape_predictor_68_face_landmarks.dat"

faceAligner = FaceAligner(FACIAL_LANDMARK_FILE)

#LOG(INFO, 'Done initializing Face Aligner object')

# ## 3. Test LeviNet model with real images

# ### 3.1 Load Cascade model for Frontal Face detection

CASCADE_FILE = os.path.join(MODEL_DIR, 'haarcascade_frontalface_alt.xml')
cascade = cv.CascadeClassifier(CASCADE_FILE)

#LOG(INFO, 'Done building Face detection model')

# ### 3.2 Helper function for Gender prediction

def getPrediction(gender_model, face_aligner, frame, faces):
    H, W = frame.shape[:2]

    pred_classes = []
    pred_probs = []
    bboxes = []
    cnt = 0
    for face in faces:
        x, y, w, h = face
        padding = w // 2

        x1 = max(0, x - padding)
        x2 = min(W, x + w + padding)
        y1 = max(0, y - 10)
        y2 = min(H, y + h + padding)

        sub_img = frame[y1:y2, x1:x2]
        bbox = (padding, 10, w, h)

        #sub_img = face_aligner.align(sub_img, bbox, size=IMAGE_WIDTH)

        sub_img = cv.resize(sub_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv.INTER_CUBIC)
        sub_img = np.expand_dims(sub_img, axis=0)

        pred = gender_model.predict(sub_img, batch_size=1)

        pred_class_idx = pred.argmax(axis=1)[0]
        pred_prob = pred[0, pred_class_idx]
        pred_classes.append('Female' if pred_class_idx == 0 else 'Male')
        pred_probs.append(pred_prob)
        bboxes.append([x1, y1, x2-x1, y2-y1] )

    return bboxes, pred_classes, pred_probs

def predictGender(gender_model, face_aligner, frame):
    faces = getFaces(cascade, frame)

    start_time = time.time()
    bboxes, pred_classes, pred_probs = getPrediction(gender_model, face_aligner, frame, faces)
    end_time = time.time()
    prx_time = end_time - start_time
    #LOG(DEBUG, 'Prediction time: {:>.3f}'.format(prx_time) )

    frame = drawFaces(frame, faces, pred_classes, pred_probs, thick=3)
    #frame = drawFaces(frame, bboxes, pred_classes, pred_probs, thick=3)

    return frame



####################################################################################################

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    i=1
    while i<10:
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
        i+=1

def get_frame():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=800, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=500, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=5, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

#    i=1
    while True:  # fps._numFrames < 120
        frame = video_capture.read()
#        frame = predictGender(genderModel, faceAligner, frame)
        input_q.put(frame)
#        input_q.put(stringData)

        t = time.time()

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        imgencode=cv2.imencode('.jpg',output_rgb)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
#        i+=1

#        cv2.imshow('Video', output_rgb)
        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()

# def get_frame():
#
#     camera_port=0
#
#     ramp_frames=100
#
#     camera=cv2.VideoCapture(camera_port) #this makes a web cam object
#
#
#     i=1
#     while True:
#         retval, im = camera.read()
#         imgencode=cv2.imencode('.jpg',im)[1]
#         stringData=imgencode.tostring()
#         yield (b'--frame\r\n'
#             b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
#         i+=1
#
#     del(camera)


CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()

@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True, threaded=True)
