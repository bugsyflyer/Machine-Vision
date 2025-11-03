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
        self.old_image = None

        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        if self.should_move:
            old_image = self.cv_image
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
        if not self.cv_image is None and self.should_move:
            # TODO: calculate and visualize the center of the "new" keypoints (most updated frame)
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

    def get_keypoints(self):
        # Create mask around area near object
        keypoint_detect_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        # wider circle with lower value corresponding to less keypoints far from the point given
        cv2.circle(keypoint_detect_mask, (30 + new_width//16, -90 + new_height//4), 400, 20, thickness=-1)
        # smaller circle with high chance of giving keypoints close to the click
        cv2.circle(keypoint_detect_mask, (30 + new_width//16, - 90 + new_height//4), 90, 255, thickness=-1)
        print(keypoint_detect_mask.shape)
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create the ORB detector object
        orb = cv2.ORB_create(nfeatures=500)

        # Detect keypoints and compute descriptors using the mask to determine where to sample keypoints
        keypoints, descriptors = orb.detectAndCompute(gray, keypoint_detect_mask)

        print("Algorithm done")

        # Draw keypoints on the image
        output_image = cv2.drawKeypoints(image, keypoints, None, color=(255, 255, 0), flags=0)

        # Display the original image with keypoints marked
        plt.figure(figsize = (10, 8))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        #plt.imshow(keypoint_detect_mask) # show where on image stuff is being selected
        plt.title('ORB Feature Detection')
        plt.show()
        print("done plotting")

        print("Starting keypoint matching")

        # Load next image
        raw_new_img = cv2.imread("../media/human_follow_3.jpg")
        assert raw_new_img is not None, "file could not be read"
        image_new = cv2.resize(raw_new_img, (new_width, new_height))

        gray_new = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)

    def visualize_keypoints(self):
        # Create keypoints on new image
        keypoints_2, descriptors_2 = orb.detectAndCompute(gray_new, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors.
        matches = bf.match(descriptors,descriptors_2)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(image,keypoints,image_new,keypoints_2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        #find center
        list_kp2 = [keypoints_2[item.trainIdx].pt for item in matches[:10]]

        points = []

        for point in list_kp2:
            points.append([point[0], point[1]])

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


        cv2.circle(img3, (image.shape[1]+int(mean_point[0]), int(mean_point[1])), 5, (0,0,255), thickness=4)

        plt.imshow(img3),plt.show()

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
