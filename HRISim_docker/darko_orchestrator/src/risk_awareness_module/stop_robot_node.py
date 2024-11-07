#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
from topic_manager import PublisherManager
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def publish_stop_message():
    rospy.init_node('reschedule_publisher', anonymous=True)
    client = actionlib.SimpleActionClient('darko/move_base', MoveBaseAction)
    client.wait_for_server()
    stop_publisher = PublisherManager("/risk_monitoring/reschedule", Bool)
    rate = rospy.Rate(1)  # 10 messaggi al secondo

    while not rospy.is_shutdown():
        stop_publisher._publish_msg(True)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_stop_message()
    except rospy.ROSInterruptException:
        pass