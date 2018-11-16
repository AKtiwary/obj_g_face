import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

from gender_classification import *

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
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    args = parser.parse_args()

    input_q = Queue(5)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

    while True:
        frame = video_capture.read()
        frame = predictGender(genderModel, faceAligner, frame)
#       end_time = time.time()
#       prx_time = end_time - start_time
#       LOG(DEBUG, 'Processing time: {:>.3f}\n'.format(prx_time))

        input_q.put(frame)

        
        #    cv.imwrite(image)
       # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']
            for point, name, color in zip(rec_points, class_names, class_colors):
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmin'] * args.width) + len(name[0]) * 6,
                               int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font,
                            0.3, (0, 0, 0), 1)
#            start_time = time.time()
            
           
            cv2.imshow('Video', frame)

        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()
