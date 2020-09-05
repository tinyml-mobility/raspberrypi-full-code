"""Example using TF Lite to classify objects with the Raspberry Pi camera."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import io
import time
import numpy as np
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


def main():

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
        elapsed_ms = (time.time() - start_time) * 1000

    #        label_id, prob = results[0]

        stream.seek(0)

        stream.truncate()

#        camera.annotate_text = '%s %.2f\n%.1fms' % (labels[label_id], prob,

#                                                   elapsed_ms)
    finally:

      camera.stop_preview()


if __name__ == '__main__':
  main()