import numpy as np
import cv2 as cv
from PIL import Image, ImageFont, ImageDraw
from utils import *

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def getFaces(cascade, image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    faces = cascade.detectMultiScale(image, scaleFactor=1.2, \
                                    minNeighbors=5, minSize=(30, 30), \
                                    flags=cv.CASCADE_SCALE_IMAGE)
    return faces

def drawFaces(image, faces, classes, probs, color=COLOR_YELLOW, thick = 2):
    if len(faces) == 0:
        return image
        exit()

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', \
                                size=np.floor(2.5e-2 * image.shape[0] + 0.5).astype('int32') )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for (x, y, w, h), class_, prob in zip(faces, classes, probs):
        # label_size = np.array([w, (draw.textsize(label, font) )[1] ] )
        label = '{}:{:>.2f}'.format(class_, prob)
        label_size = draw.textsize(label, font) 

        if y - label_size[1] >= 0:
            text_coor = np.array([x, y - label_size[1] ] )
        else:
            text_coor = np.array([x, y + 1] )

        for i in range(thick):
            draw.rectangle([x+i, y+i, x+w-i, y+h-i], outline=color)
        
        draw.rectangle([tuple(text_coor), tuple(text_coor + label_size) ], fill=color)
        draw.text(text_coor, label, fill=COLOR_BLACK, font=font)
    del draw
    return np.asarray(image)
