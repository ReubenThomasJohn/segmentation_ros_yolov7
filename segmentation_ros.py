#!/usr/bin/env python3

import this
import cv2
import rospy
from yolov7_class import Segmentation

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import numpy as np


class Detector:
    def __init__(self):
        self.loadParameters()
        self.bridge = CvBridge()
        self.rgb_image = None
        self.segmentation = Segmentation()

    def subscribeToTopics(self):
        # rospy.loginfo("Subscribed to topics")
        rospy.Subscriber(self.image_topicname, Image,
                         self.storeImage, buff_size = 2**24, queue_size=1)

    def loadParameters(self):
        '''
        do something
        '''
        self.image_topicname = "/camera/color/image_raw/compressed"
        self.pub_topic_name = "/segmentation/segmented_image"
    
    def publishToTopics(self):
        # rospy.loginfo("Published to topics")
        self.DetectionsPublisher = rospy.Publisher(
            self.pub_topic_name, Image, queue_size=1)
    
    def callObjectDetector(self, image, depth_image): # Copy for Obj Detection remove for loops
        '''
        Call the segmentation model related functions here (Reuben, Mayur)
        and the final publish function (To be done by sahil)
        '''
        self.segmentation.apply_yolo_model(image)
        output = self.segmentation.post_process()
        self.callPublisher(output)

    def callPublisher(self, image):
        '''
        the final publisher function
        '''
        image2publish = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        # if segmented_image:
            # print("Segmented")
        self.DetectionsPublisher.publish(image2publish)