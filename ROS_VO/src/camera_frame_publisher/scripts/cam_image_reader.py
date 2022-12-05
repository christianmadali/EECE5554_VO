#!/usr/bin/env python3
from __future__ import print_function

import roslib
# roslib.load_manifest('cam_image_reader')
import sys
import rospy
import rospkg
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class ros_image_converter():

    camera0_cam_info_topic = 'camera_array/cam0/camera_info'
    camera0_raw_image_topic = 'camera_array/cam0/image_raw'

    camera1_cam_info_topic = 'camera_array/cam1/camera_info'
    camera1_raw_image_topic = 'camera_array/cam1/image_raw'
    
    flir_cam_info_topic = '/flir_boson/camera_info'
    flir_cam_raw_image_topic = '/flir_boson/image_raw'

    cam0_header = 'cam_0_optical_frame'
    cam1_header = 'cam_1_optical_frame'
    flir_header = 'boson_camera'

    cv_bridge = CvBridge()

    rospack = rospkg.RosPack()
    ROS_PKG_NAME = 'camera_frame_publisher'
    ROS_PKG_PATH = rospack.get_path(ROS_PKG_NAME) 
    VIDEO_OUTPUT_PATH = ROS_PKG_PATH+'/scripts/'
    IMAGE_OUTPUT_PATH_CAM0 = ROS_PKG_PATH + '/CAM0_Dataset/'
    IMAGE_OUTPUT_PATH_CAM1 = ROS_PKG_PATH + '/CAM1_Dataset/'

    CAM0_OUTPUT_FILENAME = ROS_PKG_PATH+'CAM0.mp4'
    CAM1_OUTPUT_FILENAME = ROS_PKG_PATH+'CAM1.mp4'


    
    
    output0 = cv2.VideoWriter(CAM0_OUTPUT_FILENAME,cv2.VideoWriter_fourcc(*'mp4v'),15,(1224,1024))
    output1 = cv2.VideoWriter(CAM1_OUTPUT_FILENAME,cv2.VideoWriter_fourcc(*'mp4v'),15,(1224,1024))

    cv_image_0 = None
    cv_image_1 = None
    cv_flir_image = None
    camera_types = ['MONO','STEREO','IR']
    K0 = None
    K1 = None
    K2 = None

    D0 = None
    D1 = None
    D2 = None



    def __init__(self,camera_type = 'MONO'):
        
        self.camera0_raw_image_sub = rospy.Subscriber(self.camera0_raw_image_topic,Image,self.camera0_raw_image_callback)
        self.camera0_info_sub = rospy.Subscriber(self.camera0_cam_info_topic,CameraInfo,self.camera0_info_callback)

        self.camera1_raw_image_sub = rospy.Subscriber(self.camera1_raw_image_topic,Image,self.camera1_raw_image_callback)
        self.camera1_info_sub = rospy.Subscriber(self.camera1_cam_info_topic,CameraInfo,self.camera1_info_callback)

        self.flir_cam_raw_image_sub = rospy.Subscriber(self.flir_cam_raw_image_topic,Image,self.flir_cam_raw_image_callback)
        self.flir_cam_info_sub = rospy.Subscriber(self.camera1_cam_info_topic,CameraInfo,self.flir_cam_info_callback)

    def get_current_timestamp(self):
        currTime = rospy.Time.now()
        currTime = 1.0*currTime.secs + 1.0*currTime.nsecs/pow(10,9)
        return(currTime)
    
    def get_current_secs(self):
        currTime = rospy.Time.now()
        currTime = currTime.secs
        return currTime

    def get_current_nsecs(self):
        currTime = rospy.Time.now()
        currTime = currTime.nsecs
        return currTime

    def camera0_raw_image_callback(self,data):

        if(data.header.frame_id== self.cam0_header):
            try:
                self.cv_image_0 = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)
    
    def camera1_raw_image_callback(self,data):

        if(data.header.frame_id==self.cam1_header):
            try:
                self.cv_image_1 = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

    def flir_cam_raw_image_callback(self,data):

        if(data.header.frame_id==self.flir_header):
            try:
                self.cv_flir_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

    def camera0_info_callback(self,data):
        pass
        
    def camera1_info_callback(self,data):
        pass

    def flir_cam_info_callback(self,data):
        pass

    def get_frame(self,cam_number=0):

        if(isinstance(self.cv_image_0,np.ndarray) and isinstance(self.cv_image_1,np.ndarray) and isinstance(self.cv_flir_image,np.ndarray)):
            if cam_number<1:
                return(True,self.cv_image_0)
            elif cam_number==1:
                return(True,self.cv_image_1)
            else:
                return(True,self.cv_flir_image)
        else:
            return(False,None)
            
    def __del__(self):
        self.output0.release()
        self.output1.release()
        print('Exiting Image Converter')

def main(args):
    ic = ros_image_converter()
    rospy.init_node('image_converter', anonymous=True)
    rate = rospy.Rate(hz=60)
    img_count = 0
    try:
        while not rospy.is_shutdown():

            ret,frame = ic.get_frame(0)
            ret1,frame1 = ic.get_frame(1)
            if(ret and ret1):
                print(frame.shape)
                file_name_1 = ic.IMAGE_OUTPUT_PATH_CAM0 + 'IMAGE_'+ str(img_count) + '.png'
                file_name_2 = ic.IMAGE_OUTPUT_PATH_CAM1 + 'IMAGE_'+ str(img_count) + '.png'
                print(file_name_2)
                cv2.imshow('Window',frame)
                cv2.imwrite(file_name_1,frame)
                cv2.imwrite(file_name_2,frame1)

                img_count+=1
                if cv2.waitKey(1) == ord('q'):
                    break
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)