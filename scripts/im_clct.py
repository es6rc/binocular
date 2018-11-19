#!/usr/bin/env python

import rospy
import math
import cv2
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from im import image_converter
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

def parallelstereo():
	ic1 = image_converter('/binocular/righteye/image_raw', False)
	ic2 = image_converter('/binocular/lefteye/image_raw', False)

	# Subscribe to the state(position) of the ball
	mypause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
	myunpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
	set_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

	# Initialize rosnode
	rospy.init_node('parallelstereo', anonymous=True)
	rate = rospy.Rate(0.1) # 10hz

	# Set pose of the model
	pose = Pose()
	pose.position.x = 0.0
	pose.position.y = 1.0
	pose.position.z = 1.65
	pose.orientation.x = 0.0
	pose.orientation.y = 0.0
	pose.orientation.z = 0.0
	pose.orientation.w = 1.0

	# Set robot ought to be moved
	state = ModelState()
	state.model_name = "Checkerboard"
	state.pose = pose
	state.twist.linear.x = 0.0
	state.twist.linear.y = 0.0
	state.twist.linear.z = 0.0
	state.twist.angular.x = 0.0
	state.twist.angular.y = 0.0
	state.twist.angular.z = 0.0
	state.reference_frame = 'world'
	try:
		return_msg = set_pos(state)
		print return_msg.status_message
	except Exception, e:
		rospy.logerr('Error on Calling Service: %s', str(e))
		
	while not rospy.is_shutdown():
		for i in range(93):
			#Pause physics
			try:
				mypause()
			except Exception, e:
				rospy.logerr('Error on Calling Service: %s', str(e))

			# Renew the position of the model
			distance = (i)/10. + 1 -.3
			pose.position.y = distance
			state.pose = pose

			# Call the Service to publish position info
			#rospy.wait_for_service('/gazebo/set_model_state')
			try:
				return_msg = set_pos(state)
				print return_msg.status_message
			except Exception, e:
				rospy.logerr('Error on Calling Service: %s', str(e))

			print pose
			filename = './parallelstereo/r_%s.jpg' % distance 
			cv2.imwrite(filename, ic1.raw_image)
			filename = './parallelstereo/l_%s.jpg' % distance
			cv2.imwrite(filename, ic2.raw_image)

			# Resume physics
			try:
				myunpause()
			except Exception, e:
				rospy.logerr('Error on Calling Service: %s', str(e))

			rate.sleep()
		break


def vergencestereo():
	ic1 = image_converter('/binocular/righteye/image_raw', False)
	ic2 = image_converter('/binocular/lefteye/image_raw', False)

	# Subscribe to the state(position) of the ball
	mypause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
	myunpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
	set_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
	pub2 = rospy.Publisher('/binocular/joint2_position_controller/command', Float64, queue_size=10)
	pub4 = rospy.Publisher('/binocular/joint4_position_controller/command', Float64, queue_size=10)

	# Initialize rosnode
	rospy.init_node('vergencestereo', anonymous=True)
	rate = rospy.Rate(0.3) # 10hz

	# Set pose of the model
	pose = Pose()
	pose.position.x = 0.0
	pose.position.y = 1.0
	pose.position.z = 1.65
	pose.orientation.x = 0.0
	pose.orientation.y = 0.0
	pose.orientation.z = 0.0
	pose.orientation.w = 1.0

	# Set robot ought to be moved
	state = ModelState()
	state.model_name = "Checkerboard"
	state.pose = pose
	state.twist.linear.x = 0.0
	state.twist.linear.y = 0.0
	state.twist.linear.z = 0.0
	state.twist.angular.x = 0.0
	state.twist.angular.y = 0.0
	state.twist.angular.z = 0.0
	state.reference_frame = 'world'
	try:
		return_msg = set_pos(state)
		print return_msg.status_message
	except Exception, e:
		rospy.logerr('Error on Calling Service: %s', str(e))
		
	while not rospy.is_shutdown():
		for i in range(93):
			#Pause physics
			try:
				mypause()
			except Exception, e:
				rospy.logerr('Error on Calling Service: %s', str(e))

			# Renew the position of the model
			distance = (i)/10. + 1 -.3
			pose.position.y = distance
			state.pose = pose
			# Give command to motor for vergence
			theta = math.atan2(0.7, distance)
			pub2.publish(-theta)
			#pub3.publish(q4)
			pub4.publish(theta)

			# Call the Service to publish position info
			#rospy.wait_for_service('/gazebo/set_model_state')
			try:
				return_msg = set_pos(state)
				print return_msg.status_message
			except Exception, e:
				rospy.logerr('Error on Calling Service: %s', str(e))

			print pose
			filename = './vergencestereo/r_%s.jpg' % distance 
			cv2.imwrite(filename, ic1.raw_image)
			filename = './vergencestereo/l_%s.jpg' % distance
			cv2.imwrite(filename, ic2.raw_image)

			# Resume physics
			try:
				myunpause()
			except Exception, e:
				rospy.logerr('Error on Calling Service: %s', str(e))

			rate.sleep()
		break


if __name__ == '__main__':
	try: vergencestereo()
	except rospy.ROSInterruptException: 
		pass