import sys
import time
import argparse
import logging
import threading
import queue
import statistics

import cv2
import numpy as np

from tf_pose import common
from tf_pose.common import CocoPart
from tf_pose.estimator import TfPoseEstimator, BodyPart, Human
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
RUNNING_AVG_SIZE = 3
FACE_PART_INDEXES = [CocoPart.REye.value, CocoPart.LEye.value,
                     CocoPart.REar.value, CocoPart.LEar.value,
                     CocoPart.Nose.value]

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
    mean_parts = []
    mean_human = Human([])
    for part_index in range(CocoPart.Background.value):
        parts = []
        for human in humans:
            try:
                part = human.body_parts[part_index]
                parts.append(part)
            except KeyError:
                pass
        if len(parts) == 0:
            continue
        mean_x = statistics.mean([part.x for part in parts])
        mean_y = statistics.mean([part.y for part in parts])
        mean_part = BodyPart(f'mean-{part_index}', part_index,
                             mean_x, mean_y, None)
        mean_human.body_parts[part_index] = mean_part
    return mean_human


def get_human_size(human, img_w, img_h):
    face_box = human.get_face_box(img_w, img_h)
    if face_box is None:
        return None
    return face_box['w'] * face_box['h']


def remove_face(human):
    for part_index in FACE_PART_INDEXES:
        try:
            del human.body_parts[part_index]
        except KeyError:
            pass
    return human


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
            running_humans.append(target_human)
            if len(running_humans) > RUNNING_AVG_SIZE:
                running_humans.pop(0)
            mean_human = pose_average(running_humans)
            final_human = remove_face(mean_human)
            image = TfPoseEstimator.draw_humans(image, [final_human])
        cv2.imshow('recap', image)

    if cv2.waitKey(1) == ESC_KEY:
        break

human_thread.join()
cv2.destroyAllWindows()
