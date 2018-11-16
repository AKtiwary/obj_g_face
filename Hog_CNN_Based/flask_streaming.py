#!/usr/bin/env python
from flask import Flask, render_template, Response
import face_recognition
import glob

import cv2
import sys
import numpy

app = Flask(__name__)

known_face_encodings=[]
known_face_names=[]


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    i = 1
    while i < 10:
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + str(i) + b'\r\n')
        i += 1


def get_frame(known_face_encodings, known_face_names):
    camera_port = 0

    # ramp_frames=10

    camera = cv2.VideoCapture(camera_port)  # this makes a web cam object
    # fgbg = cv2.createBackgroundSubtractorMOG()
    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    i = 1
    while True:
        retval, im = camera.read()
        #im = cv2.resize(im1, (640, 360))
        # fgmask = fgbg.apply(im)
        rgb_im = im[:, :, ::-1]

        #face_locations = face_recognition.face_locations(rgb_im, number_of_times_to_upsample=2)
        face_locations = face_recognition.face_locations(rgb_im)

        #face_locations = face_recognition.face_locations(rgb_im, model = 'cnn')
        face_encodings = face_recognition.face_encodings(rgb_im, face_locations, num_jitters=10)
        #print(face_locations)
        #print(face_encodings)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            #print(known_face_encodings, face_encoding)

            name = "Unknown"
            #print(matches)
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(im, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(im, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(im, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        imgencode = cv2.imencode('.jpg', im)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')
        i += 1

    del (camera)

def faceTraining(imageName):
    #known_face_encodings=[]
    #known_face_names=[]
    image = face_recognition.load_image_file(imageName)

    face_encoding = face_recognition.face_encodings(image)[0]

    known_face_encodings.append(face_encoding)
    known_face_names.append(imageName[7:-4])

    return(known_face_encodings,known_face_names)

def read_image():
    #known_face_encodings=[]
    #known_face_names=[]
    images = [file for file in glob.glob("images/*.jpg")]
    for image in images:
        face_encoding_temp, known_faces_temp=faceTraining(image)
        known_face_encodings.append(face_encoding_temp[0])
        known_face_names.append(known_faces_temp[0])

    return known_face_encodings, known_face_names


@app.route('/calc')
def calc():
    return Response(get_frame(known_face_encodings,known_face_names), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(get_frame(),mimetype='multipart/byterange; boundary=frame')


if __name__ == '__main__':
    known_face_encodings, known_face_names = read_image()
    app.run(host='0.0.0.0', port = "5002", debug=True, threaded=True)
