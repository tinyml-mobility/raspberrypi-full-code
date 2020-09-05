'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import io
import time
import picamera
import pygame
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

def load_labels(path):
    
  with open(path, 'r') as f:

    return {i: line.strip() for i, line in enumerate(f.readlines())}



def set_input_tensor(interpreter, image):

  tensor_index = interpreter.get_input_details()[0]['index']

  input_tensor = interpreter.tensor(tensor_index)()[0]

  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):

  """Returns a sorted array of classification results."""

  set_input_tensor(interpreter, image)

  interpreter.invoke()

  output_details = interpreter.get_output_details()[0]

  output = np.squeeze(interpreter.get_tensor(output_details['index']))



  # If the model is quantized (uint8 data), then dequantize the results

  if output_details['dtype'] == np.uint8:

    scale, zero_point = output_details['quantization']

    output = scale * (output - zero_point)



  ordered = np.argpartition(-output, top_k)

  if (top_k==1) and (output[1] > 0.9):
    res = 1
  else:
    res = 0

  return res


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:2], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0


    print("video_src")
    print(video_src)
    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('samples/data/l$

    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)



        # 헬멧 디텍션 코드
        

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
                break # 사람이 들어왔을 때


    mp3_file = "test_tts.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file)
    labels = "labels.txt"
    model = "model_edgetpu.tflite"

    interpreter = Interpreter(model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])


    interpreter.allocate_tensors()

    _, height, width, _ = interpreter.get_input_details()[0]['shape']


    with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:

        print("before cam start")

        camera.start_preview()

        print("cam start")

        try:

            stream = io.BytesIO()

            for _ in camera.capture_continuous(

                stream, format='jpeg', use_video_port=True):

                stream.seek(0)

                image = Image.open(stream).convert('RGB').resize((width, height),

                                                                Image.ANTIALIAS)

                start_time = time.time()

                results = classify_image(interpreter, image)

                print("result:")
                print(results)

                if results==0:
                pygame.mixer.music.play()
        #         time.sleep(10)


        #        label_id, prob = results[0]

                stream.seek(0)

                stream.truncate()

    #        camera.annotate_text = '%s %.2f\n%.1fms' % (labels[label_id], prob,

    #                                                  elapsed_ms)
        finally:

            camera.stop_preview()

        # if cv.waitKey(5) == 27:
        #     break

    # print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()