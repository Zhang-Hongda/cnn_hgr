#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs import msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospkg

import tensorflow as tf
import cnn_trainer as cnn
import gestures_recorder as gr
import color_filter
import numpy as np
import cv2
import os
import time
from collections import Counter

from keras.preprocessing.image import array_to_img, img_to_array

lower_range = np.array([0, 40, 50], np.uint8)  # HSV mask
upper_range = np.array([50, 250, 255], np.uint8)  # HSV mask
gesture_list = []
gesture_dic = {"nothing": 0, "start": 1, "stop": 2, "finish": 3,
               "restart": 4, "plan": 5, "execute": 6, "planandexecute": 7}
rospack = rospkg.RosPack()
model_folder = rospack.get_path('cnn_hgr')+"/model"
labels_file_path = rospack.get_path('cnn_hgr')+"/model/labels.txt"
file = open(labels_file_path, "r")
for line in file:
    # print line,
    gesture_list.append(line[:-1])


def max_index_of(array):
    m = -1
    index = -1
    for i in range(len(array)):
        if array[i] > m:
            m = array[i]
            index = i
    return index


class cnn_hgr:
    def __init__(self):
        self.bridge = CvBridge()
        self.N = 30
        self.count = 0
        self.model = cnn.read_model(model_folder)
        self.graph = tf.get_default_graph()
        self.hg_number = 0
        self.last_number = 0
        self.image_sub = rospy.Subscriber(
            "input", Image, self.callback, queue_size=1)
        self.gesture_pub = rospy.Publisher("/gesture", msg.Int8, queue_size=1)

    def on_mouse(self, event, x, y, flag, param):
        global lower_range, upper_range
        if event == cv2.EVENT_LBUTTONDBLCLK:
            color = param[y, x]
            bgr = np.uint8([[color]])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lower_range = np.array(hsv-[10, 100, 100])
            upper_range = np.array(hsv+[10, 255, 255])

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e
        # frame = cv2.flip(frame, -1)
        width = frame.shape[1]
        hight = frame.shape[0]

        cv2.rectangle(frame, (200, 100), (400, 300), (0, 255, 0), 2)
        area = frame[100:300, 200:400]
        # extract hand using skin color
        result = color_filter.color_filter(area, lower_range, upper_range)
        # suit the image for the network: reshape, normalize
        image = cv2.resize(result, (gr.width, gr.height))
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255.0
        image = image.reshape(1, gr.width, gr.height, gr.channel)
        # use the model to predict the output
        with self.graph.as_default():
            output = self.model.predict(image)
            number = max_index_of(output[0])
            if number == self.last_number:
                cv2.putText(frame, gesture_list[number], (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                self.count += 1
            else:
                cv2.putText(frame, "Unstable", (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                self.last_number = number
                self.count = 0
        cv2.line(frame, (200, 100 + (300 - 100)*self.count/self.N),(400, 100 + (300 - 100)*self.count/self.N), (0, 255, 0), 1)
        if self.count == self.N:
            self.hg_number = self.last_number
            self.gesture_pub.publish(
                gesture_dic[gesture_list[self.hg_number]])
            self.count = 0
        cv2.putText(frame, "read: "+gesture_list[self.hg_number], (200, 100-30),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

        # display
        cv2.imshow('result', result)
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.on_mouse, param=frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(3)
        if key == ord("q"):
            rospy.signal_shutdown("Quit")


if __name__ == '__main__':
    try:
        rospy.init_node("cnn_hgr")
        rospy.loginfo("Start cnn_hgr node")
        cnn_hgr()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down cnn_hgr node."
        cv2.destroyAllWindows()
