#!/usr/bin/env python
# coding: utf-8

# # Gender Prediction
# 
# This notebook shows how to train CNN network from Gil Levi work with Adience dataset
# 
# Gil Levi paper is introduced <a href=https://ieeexplore.ieee.org/document/7301352> here</a>
# 
# Adience dataset can be found <a href=https://talhassner.github.io/home/projects/Adience/Adience-data.html> here </a>

# ## 1. Load data*
# 
# (*) Run if hdf5 file dataset does not exist, otherwise jump to Section [1. Load data from HDF5 file](#load_data)

# ### 1.1 Load Attribute lists

# In[19]:


import os
from agegender_utils import *
from utils import *

fileNames = []
ages = []
genders = []
for file in os.listdir(ATTR_DIR):
    LOG(DEBUG, 'File Name:', file)
    fileDir = os.path.join(ATTR_DIR, file)
    subFileNames, subAges, subGenders = getAttributes(fileDir)
    
    fileNames += subFileNames
    ages += subAges
    genders += subGenders
    
    LOG(DEBUG, 'No. of samples: {}'.format(len(subFileNames) ) )
    
LOG(DEBUG, 'Total data: {}\n'.format(len(fileNames) ) )

LOG(INFO, 'Done preparing attributes')


# ### 1.2 Prepare image data

# In[20]:


trainImages = getImageData(fileNames)

LOG(DEBUG, 'No. of Images:', len(trainImages) )
for image in trainImages:
    if image is None:
        LOG(DEBUG, 'None image')
print()

LOG(INFO, 'Done preparing image data')


# ### 1.2.1 Resize image data

# In[22]:


for i in range(len(trainImages) ):
    trainImages[i] = cv.resize(trainImages[i],                                 (IMAGE_WIDTH, IMAGE_HEIGHT),                                 interpolation=cv.INTER_CUBIC)
    
trainImages = np.array(trainImages)

LOG(DEBUG, 'Training shape: {}\n'.format(trainImages.shape) )

LOG(INFO, 'Done preprocessing image data')


# ### 1.2.2 Binarize gender labels

# In[24]:


import numpy as np

genderEncoded = [1 if x == 'm' else 0 for x in genders]

trainLabels = np.zeros((len(genderEncoded), NUM_CLASSES), np.uint8 )
for i, gen in enumerate(genderEncoded):
    trainLabels[i][gen] = 1

LOG(DEBUG, 'Label Shape:', trainLabels.shape)
LOG(DEBUG, '5 samples: {}\n'.format(trainLabels[:5] ) )

LOG(INFO, 'Done preparing data labels')


# ### 1.2.2 Save to HDF5 file

# In[26]:


import h5py
from agegender_utils import *
from utils import *

hf = h5py.File(IMAGE_HDF5, 'w')
hf.create_dataset('images', data=trainImages)
hf.create_dataset('labels', data=trainLabels)

hf.close()

LOG(INFO, 'Done saving HDF5 file')


# <a id='load_data'></a>
# ## 1. Load data from HDF5 file

# In[8]:


import os
from agegender_utils import *
from utils import *
import h5py

loadedHf = h5py.File(IMAGE_HDF5, 'r')
trainImages = np.array(loadedHf.get('images') )
trainLabels = np.array(loadedHf.get('labels') )

loadedHf.close()

LOG(DEBUG, 'Training shape:', trainImages.shape)
LOG(DEBUG, 'Label shape:', trainLabels.shape)
LOG(DEBUG, '3 samples: {}\n'.format(trainLabels[:3]) )

LOG(INFO, 'Done loading dataset')


# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

trainIdx = np.random.randint(trainImages.shape[0] )
LOG(DEBUG, 'Train idx:', trainIdx)
LOG(DEBUG, 'Label:', classes[trainLabels[trainIdx].argmax(axis=0) ] )
plt.imshow(trainImages[trainIdx] )
    
LOG(INFO, 'Done meaning training data')


# ## 2. Split data validation and test

# In[10]:


from sklearn.model_selection import train_test_split

ratioVal = 0.15
ratioTest = 0.15

numTest= int(len(trainImages) * ratioTest)
numVal = int(len(trainImages) * ratioVal)

LOG(DEBUG, 'No. of test:', numTest)
LOG(DEBUG, 'No. of val:', numVal)

(trainImages, valImages, trainLabels, valLabels) = train_test_split(trainImages, trainLabels,                                                                         test_size=numVal,                                                                         stratify=trainLabels)
(trainImages, testImages, trainLabels, testLabels) = train_test_split(trainImages, trainLabels,                                                                          test_size=numTest,                                                                          stratify=trainLabels)

LOG(INFO, 'Training data:   ', len(trainImages) )
LOG(INFO, 'Training shape:  ', trainImages.shape)

LOG(INFO, 'Validation data: ', len(valImages) )
LOG(INFO, 'Validation shape:', valImages.shape)

LOG(INFO, 'Test data:       ', len(testImages) )
LOG(INFO, 'Test shape:      ', testImages.shape)

print()
LOG(INFO, 'Done splitting data')


# ## 3. Load and Setup LeviNet

# In[11]:


from cnn import LeviNet
from keras.optimizers import SGD

model = LeviNet.build(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL, NUM_CLASSES)
opt = SGD(lr=1e-2, decay=0.0005, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'] )

LOG(INFO, 'Done loading and setting up LeviNet')


# In[12]:


from keras.callbacks import ModelCheckpoint

fname = os.path.join(CHECKPOINT_DIR, 'weights-{epoch:03d}-{val_loss:.4f}.hdf5')
checkpoint = ModelCheckpoint(fname, monitor='val_loss', mode='min',
                             save_best_only=True, verbose=1)
callbacks = [checkpoint]

LOG(INFO, 'Done creating Model checkpoint callback')


# ## 4. Train LeviNet

# In[13]:


H = model.fit(trainImages, trainLabels,
              validation_data=(valImages, valLabels),
              batch_size=BATCH_SIZE, 
              epochs=NUM_EPOCHS,
              callbacks=callbacks, 
              verbose=1)

LOG(INFO, 'Done training LeviNet')


# ## 5. Load trained LeviNet

# In[14]:


from keras.models import load_model

modelName = os.path.join(MODEL_DIR, 'weights-030-0.2464-20181112.hdf5')
loadedModel = load_model(modelName)
LOG(INFO, 'Done loading trained LeviNet model')


# ## 6. Evaluate trained Network

# ### 6.1 Classification report on Test set

# In[15]:


from sklearn.metrics import classification_report

#
# check with test data
#
testPreds = loadedModel.predict(testImages, batch_size=64)
testPredClasses = testPreds.argmax(axis=1)
testClasses = testLabels.argmax(axis=1)
LOG(INFO, 'Classification report:\n', 
        classification_report(testClasses, 
                              testPredClasses, 
                              target_names=classes) )

wrongTest = len(testClasses[testClasses != testPredClasses] )
LOG(INFO, 'Wrong on Training data: {}/{}\n'.format(wrongTest, testImages.shape[0]) )

print()
LOG(INFO, 'Done evaluating trained LeviNet model')


# ### 6.2 Test with Training set

# In[16]:


#
# check on training data
#
trainPreds = loadedModel.predict(trainImages, batch_size=64)
trainPredClasses = trainPreds.argmax(axis=1)
trainClasses = trainLabels.argmax(axis=1)

wrongTrain = len(trainClasses[trainClasses != trainPredClasses] )
LOG(INFO, 'Wrong on Training data: {}/{}\n'.format(wrongTrain, trainImages.shape[0]) )

LOG(INFO, 'Done testing with Training set')


# ### 6.3 Test with Validation set

# In[17]:


#
# check on validation data
#
valPreds = loadedModel.predict(valImages, batch_size=64)
valPredClasses = valPreds.argmax(axis=1)
valClasses = valLabels.argmax(axis=1)

wrongVal = len(valClasses[valClasses != valPredClasses] )
LOG(INFO, 'Wrong on Validation data: {}/{}\n'.format(wrongVal, valImages.shape[0]) )

LOG(INFO, 'Done testing with Validation set')


# ## 7. Analyze training history

# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')
plt.figure(figsize=(15, 8) )
plt.plot(np.arange(len(H.history['loss']) ),
         H.history['loss'], marker='.',
         label='Training Loss' )
plt.plot(np.arange(len(H.history['acc']) ),
         H.history['acc'],
         label='Training Accuracy' )

plt.plot(np.arange(len(H.history['val_loss']) ),
         H.history['val_loss'], marker='.',
         label='Validation Loss' )
plt.plot(np.arange(len(H.history['val_acc']) ),
         H.history['val_acc'],
         label='Validation Accuracy' )

plt.title('Gender Prediction with Adience dataset')
plt.xlabel('# of Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()


# ## 8. Summary

# In this tutorial, I tackled with Gender Prediction problem and I trained from scratch with Adience dataset. Result from Levi work is reproduced and I achieved a very high accuracy.
