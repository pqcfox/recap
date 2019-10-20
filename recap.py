import sys
import time
import argparse
import logging
import threading
import queue

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

# This work is based on the run_camera.py script incluced in
# https://github.com/ildoonet/tf-pose-estimation.

# To run the below script, first install tf-pose-estimation
# via `pip install tf-pose-estimation`.

CAMERA = 0
WIDTH, HEIGHT = 384, 384
MODEL = 'mobilenet_thin'
ESC_KEY = 27
UPSAMPLE_SIZE = 4.0

estimator = TfPoseEstimator(get_graph_path(MODEL), target_size=(WIDTH, HEIGHT))
cam = cv2.VideoCapture(CAMERA)

image = None
image_queue = queue.Queue(maxsize=1)
humans = None
humans_queue = queue.Queue(maxsize=1)

running_humans = []


def update_humans():
    while True:
        image = image_queue.get()
        humans = estimator.inference(image, resize_to_default=True,
                                     upsample_size=UPSAMPLE_SIZE)
        humans_queue.put(humans)
        time.sleep(0.001)


def pose_average(humans):
    import pdb; pdb.set_trace()
    mean_parts = []
    mean_human = Human([])
    for part_index in range(common.CocoPart.Background.value):
        if part_index not in human.body_parts.keys():
            continue
        parts = [human.body_parts[part_index] for human in humans]
        mean_x = mean([part.x for part in parts])
        mean_y = mean([part.y for part in parts])
        mean_part = BodyPart(f'mean-{part_index}', part_index,
                             mean_x, mean_y, None)
        mean_human.body_parts[part_index] = mean_part


def get_human_size(human, img_w, img_h):
    face_box = human.get_face_box(img_w, img_h)
    if face_box is None:
        return None
    return face_box['w'] * face_box['h']


human_thread = threading.Thread(target=update_humans)
human_thread.start()

cv2.namedWindow('recap', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('recap', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
        img_w, img_h = image.shape[1], image.shape[0]
        target_human = None
        target_size = 0
        for human in humans:
            size = get_human_size(human, img_w, img_h)
            if size is None:
                continue
            if size > target_size:
                target_human = human
                target_size = size
        if target_human is not None:
            image = TfPoseEstimator.draw_humans(image, [target_human], imgcopy=False)
        cv2.imshow('recap', image)

    if cv2.waitKey(1) == ESC_KEY:
        break

human_thread.join()
cv2.destroyAllWindows()
