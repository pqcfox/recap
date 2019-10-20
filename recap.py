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
q = queue.Queue()

image = None
image_lock = threading.Lock()
humans = None


def update_humans():
    while True:
        image_copy = None
        with image_lock:
            if image is not None:
                image_copy = image.copy()

        if image_copy is not None:
            humans = estimator.inference(image_copy, resize_to_default=True,
                                         upsample_size=UPSAMPLE_SIZE)
            q.put(humans)
        time.sleep(0.001)


human_thread = threading.Thread(target=update_humans)
human_thread.start()

while True:
    with image_lock:
        _, image = cam.read()

    try:
        humans = q.get(False)
        with image_lock:
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    except queue.Empty:
        pass

    with image_lock:
        cv2.imshow('Recap', image)

    if cv2.waitKey(1) == ESC_KEY:
        break

human_thread.join()
cv2.destroyAllWindows()
