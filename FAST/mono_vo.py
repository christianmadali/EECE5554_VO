import cv2
import numpy as np


class MonoVO():
    def __init__(self,img_file, pose_file, pp, focal_len, lk, min_features=2000, detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        #set vars from args
        self.file_path = img_file
        f = open(pose_file)
        self.pose = f.readlines()
        self.pp = pp
        self.focal = focal_len
        self.lk = lk
        self.min_features = min_features
        self.detector = detector

        #init other vars
        self.R = np.zeros((3,3))
        self.t = np.zeros((3,3))
        self.id = 0
        self.num_features = 0

        self.process_frame()

    def process_frame(self):
        if self.id < 2:
            self.prev_frame = cv2.imread(self.file_path + str(0).zfill(3) + '.png',0)
            self.curr_frame = cv2.imread(self.file_path + str(1).zfill(3) + '.png',0)
            self.vo()
            self.id = 2
        else:
            self.prev_frame = self.curr_frame
            self.curr_frame = cv2.imread(self.file_path + str(self.id).zfill(3) + '.png',0)
            self.vo()
            self.id += 1
    
    def vo(self):
        if self.num_features < self.min_features:
            self.pt0 = self.featureDetection(self.prev_frame)
        
        self.featureTracking()

        E,_ = cv2.findEssentialMat(self.curr_good_pts, self.prev_good_pts, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        if self.id >= 2:
            _, R, t,_ = cv2.recoverPose(E, self.prev_good_pts, self.curr_good_pts, self.R.copy(), self.t.copy(), self.focal, self.pp, None)
            absolute_scale = self.get_absolute_scale()
            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.t += absolute_scale*self.R.dot(t)
                self.R = R.dot(self.R)
        else:
            _, self.R, self.t,_ = cv2.recoverPose(E, self.prev_good_pts, self.curr_good_pts, self.R, self.t, self.focal, self.pp, None)

        self.num_features = self.curr_good_pts.shape[0]


    def featureDetection(self,img):
        pts = self.detector.detect(img)

        return np.array([p.pt for p in pts],dtype = np.float32).reshape(-1, 1, 2)
    
    def featureTracking(self):
        self.pt1, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, self.curr_frame, self.pt0, None, **self.lk)
        cv2.imshow('frame',self.prev_frame)
        self.prev_good_pts = self.pt0[status == 1]
        self.curr_good_pts = self.pt1[status == 1]
    
    def get_absolute_scale(self):
        pose = self.pose[self.id].strip().split()
        curr_x = float(pose[3])
        curr_y = float(pose[7])
        curr_z = float(pose[11])
        curr_vect = np.array(([curr_x],[curr_y],[curr_z]))
        self.true_coordinates = curr_vect

        pose = self.pose[self.id-1].strip().split()
        prev_x = float(pose[3])
        prev_y = float(pose[7])
        prev_z = float(pose[11])
        prev_vect = np.array(([prev_x],[prev_y],[prev_z]))

        return np.linalg.norm(curr_vect - prev_vect)

    def get_true_coordinates(self):
        return self.true_coordinates
    
    def get_mono_coordinates(self):
        mono_coords = np.matmul(np.identity(3)*-1,self.t)
        return mono_coords

        