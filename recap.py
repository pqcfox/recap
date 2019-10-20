import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# This work is based on the run_camera.py script incluced in
# https://github.com/ildoonet/tf-pose-estimation.

fps_time = 0

CAMERA = 0
WIDTH, HEIGHT = 368, 368
MODEL = 'mobilenet_thin'
ESC_KEY = 27

estimator = TfPoseEstimator(get_graph_path(MODEL), target_size=(WIDTH, HEIGHT))
cam = cv2.VideoCapture(args.camera)
_, image = cam.read()

while True:
    _, image = cam.read()
    humans = e.inference(image, resize_to_default=True)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    if cv2.waitKey(1) == ESC_KEY:
        break

cv2.destroyAllWindows()
