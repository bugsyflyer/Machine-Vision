import random
import rclpy
from threading import Thread
from rclpy.node import Node
import time
from neato2_interfaces.msg import Bump # detect if robot has run into something
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt

class ClickTracker(Node):
    """ The ClickTracker is a Python object that encompasses a ROS node that
        can process images from the camera and detect a (potentially) moving
        object selected through clicks. The node will issue motor commands to
        move forward while keeping the object in the center of the camera's
        field of view. The robot stops when it runs into something, likely the object it is tracking. 
        """

    def __init__(self, image_topic):
        """ Initialize the object tracker """
        super().__init__('object_tracker')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.create_subscription(Image, image_topic, self.process_image,10)
        self.bump_state = False
        self.create_subscription(Bump,"bump",self.process_bump,10) # if robot has hit something
        
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10) #command robot movement
        # where on image object is
        self.center_x = None
        self.center_y = None
        
        self.should_move = False
        self.old_image = None
        self.downscaling_factor = 4 # reduce image resolution for processing speed
        # ORB keypoints
        self.keypoints = None
        self.old_keypoints = None
        self.descriptors = None
        self.old_descriptors = None
        self.frame = False # For emulating neato's low framerate

        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.pub.publish(Twist())

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing. Save the previous image/keypoints.
            """
        # emulate low neato framerate
        if (random.randint(1,10) != 2):
            return
        print("frame")
        self.frame = True

        # downsize images
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_image = cv2.resize(self.cv_image,(self.cv_image.shape[1]//self.downscaling_factor,self.cv_image.shape[0]//self.downscaling_factor))

        if self.should_move:
            # Save image and keypoints from previous image
            self.old_image = self.cv_image
            self.old_keypoints = self.keypoints
            self.old_descriptors = self.descriptors

    def process_bump(self, msg):
        "Determine whether the robot has run into anything"
        if (msg.left_front == 1 or \
            msg.right_front == 1 or \
            msg.right_side == 1 or \
            msg.left_side == 1):
            self.bump_state = True
        

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2. 
            """
        cv2.namedWindow('video_window',cv2.WINDOW_NORMAL)
        
        cv2.setMouseCallback('video_window', self.process_mouse_event)
        while True:
            self.run_loop()
            time.sleep(0.1)

    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so the user can click on an object in the image."""
      
        # click event for controlling vehicle motion
        if event == cv2.EVENT_LBUTTONDOWN:
            self.should_move = not(self.should_move)
            if self.should_move:
                # save click location as first guess of object location
                self.center_x = x
                self.center_y = y
                print("clicked on: x:",x," y:",y)
                

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function
        if self.frame == False: # low framerate emulation
            return
        self.frame = False
        
        if (not self.cv_image is None) and self.should_move:
            # normalize self.center_x
            norm_x_pose = (self.center_x - self.cv_image.shape[1]/2) / self.cv_image.shape[1]
            # create message pose (stopped, else move towards target)
            msg_cmd = Twist()
            if self.should_move is True:
                if self.old_keypoints is not None:
                    # Do keypoint matching to update the object location guess
                    self.get_matching_keypoints(self.cv_image)
                    self.get_mean_of_keypoints()
                img_with_drawn_keypoints = self.get_surrounding_keypoints(self.cv_image)

                # Command robot movement based on x of object location
                msg_cmd.linear.x = 0.05
                msg_cmd.angular.z = -norm_x_pose/2
                if self.bump_state is True:
                    msg_cmd = Twist()
                self.pub.publish(msg_cmd)

            # Display keypoints 
            cv2.imshow('video_window', img_with_drawn_keypoints)
            cv2.waitKey(5)
        elif not self.cv_image is None:
            cv2.imshow('video_window', self.cv_image)
            cv2.waitKey(5)
            

    def get_surrounding_keypoints(self, image):
        """ Find keypoints that are near the current guess of the object on the inputted image. Return an image with these keypoints drawn on it
        """
        # Create mask around area near object
        keypoint_detect_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        # wider circle with lower value corresponding to less keypoints far from the point given
        cv2.circle(keypoint_detect_mask, (self.center_x, self.center_y), image.shape[0]//6, 30, thickness=-1)
        # smaller circle with high chance of giving keypoints close to the click
        cv2.circle(keypoint_detect_mask, (self.center_x, self.center_y), image.shape[0]//12, 255, thickness=-1)
        print(keypoint_detect_mask.shape)
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create the ORB detector object
        orb = cv2.ORB_create(nfeatures=500)

        # Detect keypoints and compute descriptors using the mask to determine where to sample keypoints
        keypoints, descriptors = orb.detectAndCompute(gray, keypoint_detect_mask)
        if descriptors is None:
            print("No keypoints detected")
            self.keypoints = self.old_keypoints
            self.descriptors = self.old_descriptors
            return image
        self.keypoints = keypoints
        self.descriptors = descriptors

        print("Algorithm done")

        # Draw keypoints on the image
        output_image = cv2.drawKeypoints(image, keypoints, None, color=(255, 255, 0), flags=0)
        return output_image
        

    def get_matching_keypoints(self, image):
        """Match the keypoints found on the provided image that were found in the previous image. Provide a list of the best matches. 
        """
        # Convert image to grayscale
        gray_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create the ORB detector object
        orb = cv2.ORB_create(nfeatures=500)

        # Create keypoints on new image
        keypoints_2, descriptors_2 = orb.detectAndCompute(gray_new, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors.
        matches = bf.match(self.descriptors,descriptors_2)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        best_keypoints = [keypoints_2[match.trainIdx] for match in matches[:60]]

        best_descriptors = [descriptors_2[match.trainIdx] for match in matches[:60]]
        self.keypoints = best_keypoints
        self.descriptors = best_descriptors
        return
    
        
    def get_mean_of_keypoints(self):
        """Find mean location of keypoints, ignoring outliers, and save as object location guess
        """
        points = []
        print(self.keypoints)

        for point in self.keypoints:
            points.append([point.pt[0], point.pt[1]])

        points_array = np.array(points)

        # median and Median Absolute Deviation
        median = np.median(points_array, axis=0)

        mad = np.median(np.abs(points_array - median), axis=0)
        threshold = 3.0

        #makes boolean array
        mask = np.all(np.abs(points_array - median) / (mad + 1e-6) < threshold, axis=1)

        filtered_points = points_array[mask]
        mean_point = np.mean(filtered_points, axis=0)
        print(mean_point)
        # Set mean point
        self.center_x = int(mean_point[0])
        self.center_y = int(mean_point[1])


# Run ROS node

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
