#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs import msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospkg

import color_filter
import numpy as np
import time
import cv2
import os
import sys
import argparse

# path to save captures
dataset_folder = '../gestures'
class_name = 'default_class_name'
file_format = 'png'

# size of image to save
width, height, channel = 32, 32, 1
grayscale = True

lower_range = np.array([0, 40, 50], np.uint8)  # HSV mask
upper_range = np.array([50, 250, 255], np.uint8)  # HSV mask


class gestures_recorder:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "input", Image, self.callback, queue_size=1)
        self.dataset_path = os.path.join(dataset_folder, class_name)
        # create path if not exists
        self.check_path(self.dataset_path)
        self.capture = False
        self.N = 0

        print("##############################")
        print("dataset folder: "+dataset_folder)
        print("class name: "+class_name)
        print("file format: "+file_format)
        print("##############################")
        print("Double click on the image to select color")
        print("Press 'r' to start/stop record")
        print("Press 'q' to quit")
        print("##############################")

    def callback(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e
        img = cv2.flip(img, 1)
        # draw rectangle for area to capture
        cv2.rectangle(img, (200, 100), (400, 300), (0, 0, 255), 2)
        area = img[100:300, 200:400]
        # extract hand using skin color
        result = color_filter.color_filter(area, lower_range, upper_range)
        # start/stop capture
        if self.capture:
            filename = self.timestamped_filename(file_format)
            pic = cv2.resize(result, (width, height))
            cv2.imwrite(os.path.join(self.dataset_path, filename), pic)
            self.N += 1

        # handle keyboard events and display
        cv2.putText(img, class_name+": "+str(self.N), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('result', result)
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.on_mouse, param=img)
        cv2.imshow('frame', img)
        key = cv2.waitKey(3)
        if key == ord("q"):
            rospy.signal_shutdown("Quit")
        elif key == ord('r'):
            self.capture = not self.capture
            print("start capture" if self.capture else "stop capture")

    def on_mouse(self, event, x, y, flag, param):
        # grab references to the global variables
        global lower_range, upper_range
        if event == cv2.EVENT_LBUTTONDBLCLK:
            color = param[y, x]
            bgr = np.uint8([[color]])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lower_range = np.array(hsv-[10, 100, 100])
            upper_range = np.array(hsv+[10, 255, 255])

    def timestamp(self):
        return int(round(time.time() * 1000))

    def check_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            rospy.loginfo(path + " has been created.")

    def timestamped_filename(self, file_format):
        return str(self.timestamp())+"."+file_format


if __name__ == '__main__':
    rospack = rospkg.RosPack()
    dataset_folder = rospack.get_path('cnn_hgr')+"/gestures"
    parse=argparse.ArgumentParser()
    parse.add_argument('-name',nargs='?',type=str,default=class_name,help="new class name, default: "+class_name)
    parse.add_argument('-format',nargs='?',type=str,default=file_format,help="picture format, default: "+file_format)
    args=vars(parse.parse_args(sys.argv[1:-1]))
    class_name=args['name']
    file_format = args['format']
    try:
        rospy.init_node("gestures_recorder")
        rospy.loginfo("Start gestures_recorder node")
        gestures_recorder()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down cv_bridge_test node."
        cv2.destroyAllWindows()
