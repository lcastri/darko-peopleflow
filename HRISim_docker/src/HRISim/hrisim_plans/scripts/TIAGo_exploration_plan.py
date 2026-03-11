import math
import os
import pickle
import random
import sys
import rospy
try:
    sys.path.insert(0, os.environ["PNP_HOME"] + '/scripts')
except:
    print("Please set PNP_HOME environment variable to PetriNetPlans folder.")
    sys.exit(1)

import pnp_cmd_ros
from pnp_cmd_ros import *
from robot_msgs.msg import BatteryStatus
from std_msgs.msg import String
import actionlib
import hrisim_util.ros_utils as ros_utils
import hrisim_util.constants as constants
import networkx as nx
from robot_srvs.srv import NewTask, FinishTask, VisualisePath
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty


def send_goal(p, next_dest, nextnext_dest=None, time_threshold=-1, first=False):
    pos = nx.get_node_attributes(G, 'pos')
    x, y = pos[next_dest]
    if nextnext_dest is not None:
        x2, y2 = pos[nextnext_dest]
        angle = math.atan2(y2-y, x2-x)
        inputs = [x, y, angle, time_threshold, 0 if first or rospy.get_param('/hrisim/robot_obs', False) else 1]
    else:
        inputs = [x, y, 0, time_threshold, 0]
    p.exec_action('gotoobs', "_".join([str(_input) for _input in inputs]))
    
    
def heuristic(a, b):
    pos = nx.get_node_attributes(G, 'pos')
    (x1, y1) = pos[a]
    (x2, y2) = pos[b]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_next_goal():
    global ROBOT_CLOSEST_WP, INB_PATH
    
    if not rospy.get_param('/hrisim/robot_busy'):     
        if rospy.get_param('/peopleflow/timeday') == constants.TOD.STARTING.value:
            return constants.WP.CORRIDOR1.value, True
                    
        else:
            if len(INB_PATH) > 0:
                rospy.logwarn("It's off time, going to clean the shop.")
                return INB_PATH.pop(0), True
            else:
                rospy.logwarn("No cleaning tasks left, shutting down the planning.")
                return None, False
        
        
def Plan(p):
    while not ros_utils.wait_for_param("/pnp_ros/ready"):
        rospy.sleep(0.1)
        
    global NEXT_GOAL, QUEUE, dynobs_remove_service, dynobs_timer_service
    
    ros_utils.wait_for_service('/hrisim/new_task')
    ros_utils.wait_for_service('/hrisim/finish_task')
    ros_utils.wait_for_service('/graph/path/show')
    ros_utils.wait_for_service('/hrisim/obstacles/remove')
    ros_utils.wait_for_service('/hrisim/obstacles/timer/off')
    ros_utils.wait_for_service('/hrisim/shutdown')

    new_task_service = rospy.ServiceProxy('/hrisim/new_task', NewTask)
    finish_task_service = rospy.ServiceProxy('/hrisim/finish_task', FinishTask)
    graph_path_show = rospy.ServiceProxy('/graph/path/show', VisualisePath)
    dynobs_remove_service = rospy.ServiceProxy('/hrisim/obstacles/remove', Empty)
    dynobs_timer_service = rospy.ServiceProxy('/hrisim/obstacles/timer/off', Empty) 
    shutdown_service = rospy.ServiceProxy('/hrisim/shutdown', Empty)
    
    
    ros_utils.wait_for_param("/peopleflow/timeday")
    rospy.set_param('/hrisim/robot_busy', False)
    PLAN_ON = True
    
    while PLAN_ON:
        rospy.logerr("Planning..")
     
        if len(QUEUE) == 0:
            NEXT_GOAL, PLAN_ON = get_next_goal()
            if NEXT_GOAL is None: continue
            QUEUE = nx.astar_path(G, ROBOT_CLOSEST_WP, NEXT_GOAL, heuristic=heuristic, weight='weight')

            rospy.logwarn("____________________________________")
            rospy.logwarn(f"New goal: {NEXT_GOAL}")
            rospy.logwarn(f"Queue: {QUEUE}")
            graph_path_show(','.join(QUEUE))

            # task_id = new_task_service(NEXT_GOAL, QUEUE).task_id
            task_id = new_task_service(NEXT_GOAL, QUEUE, 0, 0, 0, 0).task_id
            firstgoal = QUEUE[0]
        
        #! Here the goal is taken from the queue
        if not rospy.get_param('/hrisim/robot_busy') and len(QUEUE) > 0:
            next_sub_goal = QUEUE.pop(0)
            nextnext_sub_goal = QUEUE[0] if len(QUEUE) > 0 else None
            if nextnext_sub_goal is None: 
                nextnext_sub_goal = INB_PATH[0] if len(INB_PATH) > 0 else None
                
            send_goal(p, next_sub_goal, nextnext_sub_goal, first=(next_sub_goal == firstgoal))
            
            # Publish +1 when reaching the final goal
            if len(QUEUE) == 0:
                finish_task_service(task_id, constants.TaskResult.SUCCESS.value)  # 1 for success

    shutdown_service()    

                                   
def cb_battery(msg):
    global BATTERY_LEVEL, QUEUE, NEXT_GOAL, GO_TO_CHARGER
    BATTERY_LEVEL = float(msg.level.data)
    
    
def cb_robot_closest_wp(wp: String):
    global ROBOT_CLOSEST_WP
    ROBOT_CLOSEST_WP = wp.data
    
    
def cb_odom(odom: Odometry):
    v = abs(odom.twist.twist.linear.x)
    if (rospy.get_param('/hrisim/robot_obs', False) and v >= 0.5):
        dynobs_remove_service()
        rospy.set_param('/hrisim/robot_obs', False)
        dynobs_timer_service()
    
    
if __name__ == "__main__":  
    BATTERY_LEVEL = None
    ROBOT_CLOSEST_WP = None
    NEXT_GOAL = None
    QUEUE = []
    rospy.set_param('/hrisim/robot_obs', False)
    
    p = PNPCmd()
    
    g_path = ros_utils.wait_for_param("/peopleflow_pedsim_bridge/g_path")
    with open(g_path, 'rb') as f:
        G = pickle.load(f)
    INB_PATH = [constants.WP.CORRIDOR1.value, constants.WP.CORRIDOR6.value, 
                constants.WP.CORRIDOR7.value, constants.WP.CORRIDOR8.value, 
                constants.WP.CORRIDOR9.value, constants.WP.CORRIDOR10.value, 
                constants.WP.CORRIDOR14.value, constants.WP.CORRIDOR25.value,
                constants.WP.CORRIDOR15.value, constants.WP.CORRIDOR20.value, 
                constants.WP.CORRIDOR19.value, constants.WP.CORRIDOR23.value,
                constants.WP.CORRIDOR22.value, constants.WP.CORRIDOR24.value, 
                constants.WP.CORRIDOR21.value, constants.WP.CORRIDOR18.value,
                constants.WP.CORRIDOR17.value, constants.WP.CORRIDOR26.value, 
                constants.WP.CORRIDOR13.value, constants.WP.CORRIDOR25.value,
                constants.WP.CORRIDOR12.value, constants.WP.ROOM2.value, 
                constants.WP.CORRIDOR16.value, constants.WP.CORRIDOR5.value,
                constants.WP.CORRIDOR4.value, constants.WP.CORRIDOR3.value, 
                constants.WP.CORRIDOR11.value, constants.WP.CORRIDOR2.value, 
                constants.WP.ROOM1.value]
    rospy.Subscriber("/hrisim/robot_battery", BatteryStatus, cb_battery)
    rospy.Subscriber("/hrisim/robot_closest_wp", String, cb_robot_closest_wp)
    rospy.Subscriber("/mobile_base_controller/odom", Odometry, cb_odom)

    p.begin()

    Plan(p)

    p.end()