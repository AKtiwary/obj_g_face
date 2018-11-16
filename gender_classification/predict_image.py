#!/usr/bin/env python
# coding: utf-8

# ## 1. Load trained LeviNet

# In[2]:


import os
from agegender_utils import *
from face_utils import *
from utils import *
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import load_model

modelName = os.path.join(MODEL_DIR, 'weights-033-0.2842-20181112.hdf5')
genderModel = load_model(modelName)

LOG(INFO, 'Done loading trained LeviNet model')


# ## 2. Load Face Aligner for Face alignment

# In[3]:


from facealigner import FaceAligner

faceAligner = FaceAligner(FACIAL_LANDMARK_FILE)

LOG(INFO, 'Done initializing Face Aligner object')


# ## 3. Test LeviNet model with real images

# ### 3.1 Load Cascade model for Frontal Face detection

# In[4]:


CASCADE_FILE = os.path.join(MODEL_DIR, 'haarcascade_frontalface_alt.xml')
cascade = cv.CascadeClassifier(CASCADE_FILE)

LOG(INFO, 'Done building Face detection model')


# ### 3.2 Helper function for Gender prediction

# In[5]:


def getPrediction(gender_model, face_aligner, image, faces):
    H, W = image.shape[:2]
    
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
        
        sub_img = image[y1:y2, x1:x2]
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


# In[6]:


def predictGender(gender_model, face_aligner, image):
    faces = getFaces(cascade, image)

    start_time = time.time()
    bboxes, pred_classes, pred_probs = getPrediction(gender_model, face_aligner, image, faces)
    end_time = time.time()
    prx_time = end_time - start_time
    #LOG(DEBUG, 'Prediction time: {:>.3f}'.format(prx_time) )
    
    image = drawFaces(image, faces, pred_classes, pred_probs, thick=3)
    #image = drawFaces(image, bboxes, pred_classes, pred_probs, thick=3)
    
    return image


# ### 3.3 Test with an image

# In[12]:


import time

#image_file = 'samples/good-place001.jpg'
image_file = 'samples/1.jpg'
image = cv.imread(image_file)
if image is None:
    LOG(ERROR, 'Image does not exist')
else:  
    start_time = time.time()
    image = predictGender(genderModel, faceAligner, image)
    end_time = time.time()
    prx_time = end_time - start_time
    LOG(DEBUG, 'Processing time: {:>.3f}\n'.format(prx_time) )

    cv.imwrite(image_file, image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 15) )
    plt.imshow(image)

print()
LOG(INFO, 'Done testing with some samples')


# In[ ]:




