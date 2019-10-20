import sys
import time
import argparse
import logging
import threading
import queue

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# This work is based on the run_camera.py script incluced in
# https://github.com/ildoonet/tf-pose-estimation.

# To run the below script, first install tf-pose-estimation
# via `pip install tf-pose-estimation`.

CAMERA = 0
WIDTH, HEIGHT = 368, 368
MODEL = 'mobilenet_thin'
ESC_KEY = 27
UPSAMPLE_SIZE = 4.0

estimator = TfPoseEstimator(get_graph_path(MODEL), target_size=(WIDTH, HEIGHT))
cam = cv2.VideoCapture(CAMERA)

image = None
image_queue = queue.Queue(maxsize=1)
humans = None
humans_queue = queue.Queue(maxsize=1)


def update_humans():
    while True:
        image = image_queue.get()
        humans = estimator.inference(image, resize_to_default=True,
                                     upsample_size=UPSAMPLE_SIZE)
        humans_queue.put(humans)
        time.sleep(0.001)


human_thread = threading.Thread(target=update_humans)
human_thread.start()

while True:
    _, image = cam.read()
    try:
        image_queue.get(None)
    except queue.Empty:
        pass
    image_queue.put(image.copy())

    try:
        humans = humans_queue.get(False)
    except queue.Empty:
        pass

    if humans is not None:
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.imshow('Recap', image)

    if cv2.waitKey(1) == ESC_KEY:
        break

human_thread.join()
cv2.destroyAllWindows()
