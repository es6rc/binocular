#!/usr/bin/env python

# import roslib; roslib.load_manifest('gazebo')
import rospy
import cv2
import sys
import numpy as np
from im import image_converter
from gazebo_msgs.srv import GetModelState
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from ik import *



# WINDOW_NAME1 = 'righteye tracking'
# WINDOW_NAME2 = 'lefteye tracking'


def trk(args):
    # Track images from two eyes
    ic1 = image_converter('/binocular/righteye/image_raw', True)
    ic2 = image_converter('/binocular/lefteye/image_raw',True)
    
    # Subscribe to the state(position) of the ball
    mypause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    myunpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    modelstate = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    #Define publishers for each joint position controller commands.
    pub0 = rospy.Publisher('/binocular/joint0_position_controller/command', Float64, queue_size=10)
    #pub1 = rospy.Publisher('/binocular/joint1_position_controller/command', Float64, queue_size=10)
    pub2 = rospy.Publisher('/binocular/joint2_position_controller/command', Float64, queue_size=10)
    pub3 = rospy.Publisher('/binocular/joint3_position_controller/command', Float64, queue_size=10)
    pub4 = rospy.Publisher('/binocular/joint4_position_controller/command', Float64, queue_size=10)
    pub5 = rospy.Publisher('/binocular/joint5_position_controller/command', Float64, queue_size=10)

    rospy.init_node('image_converter', anonymous=True)
    rate = rospy.Rate(20) # 10hz
    #print(detected)   
    while not rospy.is_shutdown():    
        v_concat = np.concatenate((ic1.image_track, ic2.image_track), axis=0)
        #v_size = np.shape(v_concat)
        #v_concat = cv2.resize(v_concat,(400,480))
        cv2.imshow('TRACKING', v_concat)
        # cv2.imshow(WINDOW_NAME2, ic2.image_track)
        cv2.waitKey(3)

        #print(pos.x, pos.y, pos.z)
        #print(ic1.detected, ic2.detected)
        #Pause physics
        try:
            mypause()
            detected = ic1.detected and ic2.detected 
        except Exception, e:
            rospy.logerr('Error on Calling Service: %s', str(e))
        #try:
        if detected:
            model = modelstate('ball','')
            pos = model.pose.position
            q1, q3, q4 = ik(pos.x, pos.y, pos.z)
            # q's are computed in their cooresponding coordinate frames
            # q3<0 as left pan coordinate
            #print(q1, q3, q4)
            #pub0.publish(q1)
            
            pub2.publish(q3)
            #pub3.publish(q4)
            pub4.publish(-q3)
            #pub5.publish(q4)
        #except Exception, e:
        #    rospy.logerr('Error on Calling Service: %s', str(e))

        # Resume physics
        try:
            myunpause()
        except Exception, e:
            rospy.logerr('Error on Calling Service: %s', str(e))
        
        rate.sleep()
    '''
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    '''
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        trk(sys.argv)
    except rospy.ROSInterruptException:
        pass
    #main(sys.argv)