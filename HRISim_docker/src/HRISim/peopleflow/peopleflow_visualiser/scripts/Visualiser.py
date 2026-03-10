#!/usr/bin/env python

import rospy
import pickle
from pedsim_srvs.srv import GetNextDestination, GetNextDestinationResponse
import hrisim_util.ros_utils as ros_utils
from robot_srvs.srv import VisualisePath


class PeopleFlowVisualiser():
    def __init__(self):
        
        rospy.Service('get_next_destination', GetNextDestination, self.dummy_get_next_dest)
        rospy.loginfo('ROS service /get_next_destination advertised')      

             
    def dummy_get_next_dest(self, req):
        pass


  
if __name__ == '__main__':
    rospy.init_node('peopleflow_pedsim_bridge')
    rate = rospy.Rate(10)  # 10 Hz
    
    pv = PeopleFlowVisualiser()
    
    try:
        g_path = str(ros_utils.wait_for_param("~g_path", timeout=10))
        with open(g_path, 'rb') as f:
            G = pickle.load(f)
            ros_utils.load_graph_to_rosparam(G, "/peopleflow/G")
            
            # Create a handle for the Trigger service
            rospy.wait_for_service('/graph/path/show')
            graph_path_show = rospy.ServiceProxy('/graph/path/show', VisualisePath)        # Call the service
            graph_path_show("")
    except Exception as e:
        rospy.logerr("Error loading graph: %s", e)
                
    rospy.logwarn("PeopleFlow Visualiser started!")
                
    rospy.spin()