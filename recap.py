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
HSE_ANGLES = [np.pi/6, np.pi/6, np.pi/6]
HSE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

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


def get_angle(human, a_part_index, b_part_index, c_part_index):
    try:
        a_part = human.body_parts[a_part_index.value]
        b_part = human.body_parts[b_part_index.value]
        c_part = human.body_parts[c_part_index.value]
    except KeyError:
        return None
    a_to_b = (a_part.x - b_part.x, a_part.y - b_part.y)
    c_to_b = (c_part.x - b_part.x, c_part.y - b_part.y)
    dot_prod = a_to_b[0] * c_to_b[0] + a_to_b[1] * c_to_b[1]
    ab_len = np.sqrt(a_to_b[0]**2 + a_to_b[1]**2)
    cb_len = np.sqrt(c_to_b[0]**2 + c_to_b[1]**2)
    angle = np.arccos(dot_prod / (ab_len * cb_len))
    return angle


def draw_target(image, human, dist_part_index, pivot_part_index,
                angles, colors, reverse_angles=False, init_angle=np.pi/2.0):
    img_h, img_w = image.shape[:2]
    try:
        dist_part = human.body_parts[dist_part_index.value]
        pivot_part = human.body_parts[pivot_part_index.value]
    except KeyError:
        return image

    distance = int(np.sqrt(((dist_part.x - pivot_part.x) * img_w)**2 +
                           ((dist_part.y - pivot_part.y) * img_h)**2))
    center = (int(pivot_part.x * img_w + 0.5),
              int(pivot_part.y * img_h + 0.5))
    axes = (distance, distance)

    target_image = image.copy()
    prev_end_angle = init_angle * 360/(2 * np.pi)
    for angle, color in zip(angles, colors):
        if reverse_angles:
            angle *= -1
        start_angle = prev_end_angle
        end_angle = int(start_angle + angle * 360/(2 * np.pi))
        prev_end_angle = end_angle
        cv2.ellipse(target_image, center, axes, 0, start_angle, end_angle, color, thickness=-1)

    image = cv2.addWeighted(image, 0.1, target_image, 0.9, 0.0)
    return image


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
            print(get_angle(final_human, CocoPart.LHip,
                            CocoPart.LShoulder, CocoPart.LElbow))

            image = draw_target(image, final_human, CocoPart.LHip, CocoPart.LShoulder,
                                HSE_ANGLES, HSE_COLORS, reverse_angles=True)
            image = TfPoseEstimator.draw_humans(image, [final_human])
        cv2.imshow('recap', image)

    if cv2.waitKey(1) == ESC_KEY:
        break

human_thread.join()
cv2.destroyAllWindows()
