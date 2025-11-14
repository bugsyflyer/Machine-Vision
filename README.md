# Machine-Vision

_ENGR3590: A Computational Introduction to Robotics_

_Franklin Noble, Brenna O'Donnell_

### Description: 
Inspired by point-and-click games, we aimed to make a project where a user could click somewhere on the current image from the neato's camera feed, and have the neato follow that object to an arbitrary stopping point. We intended to use openCV to do this and learn about different segmentation algorithms and openCV-ROS2 integration. How our method works, is the user clicks on a location in the most recent image from the Neato's camera. From this click, keypoints are generated on the image within a certain area of the click. The mean of those keypoints is found, which will become the point which the neato will start to turn/drive toward. Once the neato moves a certain amount, new keypoints will be generated evenly across the image and compared to the past ones. Matching points will be kept and used to generate a new mean. This will repeat until the neato is "close enough" to the target object.

## Choosing an algorithm (Design Desicions):
There are MANY ways to segment an image, and throughout this project we checked out or tried a whole bunch of them. 
At first we wanted a click to isolate a section of the image using grap-cut Semantic Segmentation, an algorithm to iteratively select objects based on user-drawn masks. This method took to long to compute a single frame, and did not work very well in the classroom due to all the tables and chairs.
Similar to this was a Click-based Segmentation Implementation from a paper with code in a github repo. This seemed perfect. The user could click in one location and the object the user clicked on would be isolated. THe user could click more times to make the isolated object more accurate. Unfortunately, this repo was made using Python2 and was not viable.
There were also a liteny of semantic segmentation algorithms we found that used Pytorch, which does not play well with ROS2.
We decided on Keypoint Matching with an OpenCV ORB implementation.

## Challenges:
Our first and most pervasive issue was choosing an algorithm. It was extremely time consuming and often demoralizing. We would get attached to a method or algorithm only to find that it wouldn't work for one reason or another. We fully implemented multiple algorithms just for then to not work right for whatever reason. We had less time than most on this project due to FWOP, so every failed implementation drew us closer and closer to a project disaster. We have also had trouble committing to an alorithm because we got so used to the search-and-fail loop.
Our other major challenge was scheduling. We have rather conflicting schedules which led us to only meeting in class or for multiple hours straight very infrequently. We are both in FWOP doing Set and Tech week, which meant that not only would we be unavailable most hours during tech week, but for the prior week as well. We tried to deal with this by working a ton on the project right from the outset, meeting as often and as long as we possible could. While we made a great deal of progress before FWOP really started, searching for an algorithm took a lot longer than intended and we had hoped to finish the whole project other than the write up before tech week.

##Future Steps and Lessons:
Our method is not very accurate. It just gets keypoints in the area near a click and does not determine if the keypoints are on the "correct" object and so the neato follows a general area rather than a specific semantic object. We also spent very little time debugging so our project almost certainly does not work quite as intended.
