import dlib
from utils import *

def rectToRectangle(rect):
    x, y, w, h = rect
    return dlib.rectangle(x, y, x+w, y+h)

class FaceAligner:
    def __init__(self, model):
        self.__predictor = dlib.shape_predictor(model)
#        LOG(DEBUG, 'Done loading shape predictor')

    def __findLandmarks(self, frame, rect):
        return self.__predictor(frame, rect)
    
    def align(self, frame, bbox, size=160):
        rect = rectToRectangle(bbox)
        landmark = self.__findLandmarks(frame, rect) 
        aligned_image = dlib.get_face_chip(frame, landmark, size=size)
        return aligned_image 
