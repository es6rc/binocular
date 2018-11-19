#!/usr/bin/env python

import rospy
import math

from std_msgs.msg import Float64
from std_srvs.srv import Empty

from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

def main():

    # Initiate node for allocating a model at a certain position
    rospy.init_node('set_model_position',anonymous=True)
    # Setup Services for pausing & resuming gazebo physics 
    # and change model states
    mypause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    myunpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    set_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    # Set pose of the model
    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 2.0
    pose.position.z = 1.65
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    
    # Set robot ought to be moved
    state = ModelState()
    state.model_name = "ball"
    state.pose = pose
    state.twist.linear.x = 0.0
    state.twist.linear.y = 0.0
    state.twist.linear.z = 0.0
    state.twist.angular.x = 0.0
    state.twist.angular.y = 0.0
    state.twist.angular.z = 0.0
    state.reference_frame = 'world'

    rate = rospy.Rate(2) # 25hz
    i = 0
    while not rospy.is_shutdown():
        #Pause physics
        try:
            mypause()
        except Exception, e:
            rospy.logerr('Error on Calling Service: %s', str(e))

        # Renew the position of the model
        pose.position.z = 1.65 + math.sin(i/3.)
        pose.position.x = math.cos(i/3.)
        state.pose = pose

        # Call the Service to publish position info
        #rospy.wait_for_service('/gazebo/set_model_state')
        try:
            return_msg = set_pos(state)
            # print return_msg.status_message
        except Exception, e:
            rospy.logerr('Error on Calling Service: %s', str(e))

        # Resume physics
        try:
            myunpause()
        except Exception, e:
            rospy.logerr('Error on Calling Service: %s', str(e))
        #print (i+1.)/10.
        i = i + 1
        rate.sleep()

if __name__ == '__main__':
    try: main()
    except rospy.ROSInterruptException: 
        pass