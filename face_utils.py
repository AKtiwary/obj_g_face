import numpy as np
import cv2 as cv
from PIL import Image, ImageFont, ImageDraw
from utils import *

########################################

from termcolor import colored
import numpy as np
import cv2 as cv
import time


# colors for drawing
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_LIGHT_SKY_BLUE = (250, 206, 135)
COLOR_DEEP_SKY_BLUE = (255, 191, 0)
COLOR_PURPLE = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# log mode
NONE = 0
INFO = 1
DEBUG = 2
BEGIN = 4
END = 8
ERROR = 16

log_map = {
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    BEGIN: 'BEGIN',
    END: 'END',
    ERROR: 'ERROR'
}

# LOG_ENABLE = NONE
# LOG_ENABLE = INFO
# LOG_ENABLE = INFO | ERROR
LOG_ENABLE = INFO | BEGIN | END | DEBUG | ERROR

def showImage(title, image):
    cv.imshow(title, image)

def waitImage():
	key = cv.waitKey(0)
	if key == 27 or key == ord('q'):
		pass

# extract file name from a given file path
def getFileName(file_dir):
    start_idx = file_dir.rfind('/')
    end_idx = file_dir.rfind('.')
    if end_idx == -1:
        end_idx = len(file_dir)

    return file_dir[start_idx+1 : end_idx]

# log function used to print log while running program
def LOG(log_mode, log_txt, obj=None):
    if LOG_ENABLE & log_mode:
        if obj is not None:
            log_str = '{} {}'.format(log_txt, obj)
        else:
            log_str = '{}'.format(log_txt)

        color = None
        if log_mode == INFO:
            color = 'blue'
        elif log_mode == ERROR:
            color = 'red'
        elif log_mode == BEGIN or log_mode == END:
            color = 'white'
        elif log_mode == DEBUG:
            color = 'yellow'

        print('{}'.format(colored('['+log_map[log_mode]+']', \
                            color=color, 
                            attrs=['bold'] ) ), \
                            log_str)
    else:
        pass

# measure run time with a given number of iterations
def testTime(func, args, iters=100000):
    start_time = time.time()

    for i in range(iters):
        func(args)
    
    end_time = time.time()
    
    return (end_time - start_time)


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

def drawFaces(frame, faces, classes, probs, color=COLOR_YELLOW, thick = 2):
    if len(faces) == 0:
        return frame
        exit()

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', \
                                size=np.floor(2.5e-2 * frame.shape[0] + 0.5).astype('int32') )
    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
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
    return np.asarray(frame)
