import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist

class ClickTracker(Node):
    """ The ClickTracker is a Python object that encompasses a ROS node that
        can process images from the camera and detect a (potentially) moving
        object selected through clicks. The node will issue motor commands to
        move forward while keeping the object in the center of the camera's
        field of view.
        """

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        super().__init__('ball_tracker')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.create_subscription(Image, image_topic, self.process_image, 10)
        
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.center_x = 0.
        self.center_y = 0.
        self.should_move = False

        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2. 
            """
        cv2.namedWindow('video_window')

        # Reference: How to add a slider bar
        # needs a callback function like self.set_red_lower_bound to change value of slider
        # cv2.createTrackbar('red lower bound', 'binary_window', self.red_lower_bound, 255, self.set_red_lower_bound)
        
        cv2.setMouseCallback('video_window', self.process_mouse_event)
        while True:
            self.run_loop()
            time.sleep(0.1)

    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so that you can see the color values
            associated with a particular pixel in the camera images """
      
        # click event for controlling vehicle motion
        if event == cv2.EVENT_LBUTTONDOWN:
            self.should_move = not(self.should_move)
            # TODO: Save position of mouse click

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        if not self.cv_image is None:
            # TODO: Get mean/center stuff here
            # call function, passing in image
            # set the center
            self.center_x, self.center_y = 20, 60
            # normalize self.center_x
            norm_x_pose = (self.center_x - self.cv_image.shape[1]/2) / self.cv_image.shape[1]
            # create message pose (stopped, else move towards target)
            msg_cmd = Twist()
            if self.should_move is True:
                msg_cmd.linear.x = 0.1
                msg_cmd.angular.z = -norm_x_pose
                self.pub.publish(msg_cmd)
            cv2.imshow('video_window', self.cv_image)
            cv2.waitKey(5)

if __name__ == '__main__':
    node = ClickTracker("/camera/image_raw")
    node.run()


def main(args=None):
    rclpy.init()
    n = ClickTracker("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
