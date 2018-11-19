#!/usr/bin/env python

import rospy
import math

from std_msgs.msg import Float64
from math import sin,cos,atan2,sqrt,fabs

#Define a  joint positions publisher for joint controllers.
def binocular_joint_positions_publisher():

    #Initiate node for controlling joints positions.
    rospy.init_node('joint_positions_node', anonymous=True)

    #Define publishers for each joint position controller commands.
    pub0 = rospy.Publisher('/binocular/joint0_position_controller/command', Float64, queue_size=10)
    #pub1 = rospy.Publisher('/binocular/joint1_position_controller/command', Float64, queue_size=10)
    pub2 = rospy.Publisher('/binocular/joint2_position_controller/command', Float64, queue_size=10)
    #pub3 = rospy.Publisher('/binocular/joint3_position_controller/command', Float64, queue_size=10)
    #pub4 = rospy.Publisher('/binocular/joint4_position_controller/command', Float64, queue_size=10)
    #pub5 = rospy.Publisher('/binocular/joint5_position_controller/command', Float64, queue_size=10)

    rate = rospy.Rate(50) #100 Hz

    #While loop to have joints follow a certain position, while rospy is not shutdown.
    i = 0
    while not rospy.is_shutdown():

        #Have each joint follow a sine movement of sin(i/100).
        sine_movement = sin(i/100.)

        #Publish the same sine movement to each joint.
        pub0.publish(sine_movement)
        pub2.publish(-sine_movement)
        #pub4.publish(-sine_movement)

        i = i+1 #increment i

        rate.sleep() #sleep for rest of rospy.Rate(100)

#Main section of code that will continuously run unless rospy receives interuption (ie CTRL+C)
if __name__ == '__main__':
    try: binocular_joint_positions_publisher()
    except rospy.ROSInterruptException: 
        pass