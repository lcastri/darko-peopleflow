#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point,Quaternion,Pose,PoseStamped
from std_msgs.msg import Bool,Int64,String,Float64MultiArray,Int64MultiArray
from darko_orchestrator.msg import CurrentAction, Action, State

class PublisherManager(object):
    def __init__(self,topic,msg_type,latch_bool=False):
        self._topic = topic
        self._msg_type = msg_type
        self._latch_bool = latch_bool
        self._pub = rospy.Publisher(self._topic,self._msg_type,latch=self._latch_bool,queue_size=10)

    def _publish_msg(self,msg):
        while self._pub.get_num_connections() < 1:
            rospy.sleep(.01)
        self._pub.publish(msg) 

class SubscriberManager(object):
    def __init__(self,topic,msg_type,wait_for_first_msg=False):
        self._topic = topic
        self._msg_type = msg_type
        self._wait_for_first_msg = wait_for_first_msg
        self._data = None
        self._header = None
        self._sub = rospy.Subscriber(self._topic,self._msg_type,self._cb_fun,queue_size=1 )
        if self._wait_for_first_msg:
            while self._data is None and not rospy.is_shutdown():
                rospy.sleep(0.1)
        else:
            self._reset_data()

    def _cb_fun(self, data):
        if self._msg_type in [Bool,Int64,String]:
            self._data = data.data
        elif self._msg_type in [Point,Pose]:
            self._data = data
        elif self._msg_type == PoseStamped:
            self._header = data.header
            self._data   = data.pose
        elif self._msg_type == CurrentAction:
            self._data = data
        elif self._msg_type == State:
            self._data = data
        elif self._msg_type == Float64MultiArray:
            self._data = data.data
        elif self._msg_type == Int64MultiArray:
            self._data = data.data
    
    def _reset_data(self):
        if self._msg_type == Bool:
            self._data = False
        elif self._msg_type == Int64:
            self._data = -100
        elif self._msg_type == String:
            self._data = "_"
        elif self._msg_type == Point:
            self._data = Point(-100,-100,-100)
        elif self._msg_type in [Pose,PoseStamped]:
            self._header = None
            self._data = Pose(Point(-100,-100,-100),Quaternion(-1,0,0,0))
        elif self._msg_type == CurrentAction:
            action1 = Action(action=[])
            action2 = Action(action=[])
            self._data = CurrentAction(first_action=action1, second_action=action2)
        elif self._msg_type == State:
            self._data = State()
        elif self._msg_type == Float64MultiArray:
            self._data = []
        elif self._msg_type == Int64MultiArray:
            self._data = []

    def _check_empty_data(self):
        tol = 1e-6
        if self._msg_type == Bool:
            return self._data == False
        if self._msg_type == Int64:
            return -100-tol <= self._data <= -100+tol 
        if self._msg_type == String:
            return self._data == "_"
        if self._msg_type == Point:
            xp_check = -100-tol <= self._data.x <= -100+tol
            yp_check = -100-tol <= self._data.y <= -100+tol
            zp_check = -100-tol <= self._data.z <= -100+tol
            return (xp_check & yp_check & zp_check) 
        if self._msg_type in [Pose,PoseStamped]:
            xp_check = -100-tol <= self._data.position.x <= -100+tol
            yp_check = -100-tol <= self._data.position.y <= -100+tol
            zp_check = -100-tol <= self._data.position.z <= -100+tol
            p_check = xp_check & yp_check & zp_check
            xo_check = -1-tol <= self._data.orientation.x <= -1+tol
            yo_check =  0-tol <= self._data.orientation.y <=  0+tol
            zo_check =  0-tol <= self._data.orientation.z <=  0+tol
            wo_check =  0-tol <= self._data.orientation.w <=  0+tol
            o_check = xo_check & yo_check & zo_check & wo_check
            return p_check & o_check
        if self._msg_type == Int64MultiArray:
            return self._data in [None, []]
        if self._msg_type == State:
            return self._data.state == []
