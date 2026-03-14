#!/usr/bin/env python

import subprocess

import rospy
import signal
import hrisim_util.ros_utils as ros_utils
import hrisim_util.constants as constants
from peopleflow_msgs.msg import Time as pT

TOPICS = [
    "/map",
    "/tf",
    "/tf_static",
    "/robot_pose",
    "/mobile_base_controller/odom",
    "/move_base/goal",
    "/pedsim_simulator/simulated_agents",
    "/peopleflow/counter",
    "/peopleflow/time",
    "/hrisim/robot_battery",
    "/hrisim/robot_closest_wp",
    "/hrisim/robot_tasks_info",
    "/hrisim/robot_human_collision",
    # "/hrisim/robot_clearing_distance",
    "/hrisim/robot_obs"
]


def cb_time(t: pT):
    global TIME
    TIME = t.time_of_the_day.data

   
if __name__ == '__main__':
    TIME = None

    rospy.init_node('hrisim_recording')
    rate = rospy.Rate(10)  # 10 Hz
    
    EXP = str(rospy.get_param("~bagname"))
    DTOD = str(rospy.get_param("~desired_tod"))
    
    rospy.Subscriber("/peopleflow/time", pT, cb_time)

    written = False
    while TIME is None or TIME != DTOD:
        if not written:
            rospy.logwarn(f"Waiting for the desired time of day ({DTOD}) to start recording...")
            written = True
        rate.sleep()
    rospy.logwarn(f"Desired time of day ({DTOD}) reached => Start recording")
        
    try:
        bag_process = subprocess.Popen(['rosbag', 'record', '-O', f'/home/hrisim/shared/{EXP}.bag'] + TOPICS, shell=False)
    except Exception as e:
        rospy.logerr(f"Failed to start ROS bag recording: {str(e)}")
        
    
    def shutdown_hook():
        if bag_process is not None:
            try:
                rospy.loginfo("Stopping the ROS bag recording...")
                bag_process.send_signal(signal.SIGINT) # Send SIGINT to the process (equivalent to pressing Ctrl+C)
                bag_process.wait() # Wait for the process to terminate
                rospy.loginfo("ROS bag recording stopped.")
            except Exception as e:
                rospy.logerr(f"Failed to stop ROS bag recording: {str(e)}")
                
    # Register the shutdown hook
    rospy.on_shutdown(shutdown_hook)             

    rospy.spin()