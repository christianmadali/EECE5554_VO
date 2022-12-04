from mono_vo import MonoVO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

img_path = '/home/christianmadali/EECE5554_personal/final_project/image_l/'
pose_path = '/home/christianmadali/EECE5554_personal/final_project/dataset/poses/00.txt'

focal_len = 1886.9232144
pp = (604.72148787, 493.466267)

res_t = np.zeros((3, 3))
res_R = np.zeros((3, 3))

lk = dict( winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

vo = MonoVO(img_path, pose_path, pp, focal_len, lk)

frame_count = 0
# Iterate directory
for path in os.listdir(img_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(img_path, path)):
        frame_count += 1

traj = np.zeros(shape=(600, 800, 3))
while (vo.id < frame_count):
    frame = vo.curr_frame
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

    vo.process_frame()

    true_coordinates = vo.get_true_coordinates()
    mono_coordinates = vo.get_mono_coordinates()

    true_x, true_y, true_z = [int(iter) for iter in true_coordinates]

    mono_x, mono_y, mono_z = [int(iter) for iter in mono_coordinates]

    #traj = cv2.circle(traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
    traj = cv2.circle(traj, (mono_x + 400, mono_z + 100), 1, list((0, 255, 0)), 4)

    cv2.imshow('trajectory', traj)
