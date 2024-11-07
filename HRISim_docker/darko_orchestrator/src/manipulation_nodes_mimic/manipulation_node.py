#!/usr/bin/env python3
#%%
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool,String,Int64
from topic_manager import SubscriberManager,PublisherManager
from subscribers import AmclPoseManager
import json 
import math 
import numpy as np

if __name__ == '__main__':

    rospy.init_node('fake_manipulation_node', anonymous=True)

    ### <---------- publishers  ----------> ###
    manipulation_done_pub    = PublisherManager("/manipulation/action_finished", Bool)
    manipulation_success_pub = PublisherManager("/manipulation/action_success" , Bool)


    ### <---------- subscribers ----------> ###
    manipulation_action_type_sub = SubscriberManager("/manipulation/action_type",String,False)
    target_object_sub            = SubscriberManager("/manipulation/object_type",String,False)
    target_tray_sub              = SubscriberManager("/manipulation/tray_type"  ,String,False)
    
    position_subscriber = AmclPoseManager()


    # load static data
    path_to_static_data = "../static_data"

    static_data_names = ["location_coordinates",
                            "objects_box"]

    static_data = {}

    for static_data_name in static_data_names:
        with open(path_to_static_data+"/"+static_data_name+".json", "r") as json_file:
            static_data[static_data_name] = json.load(json_file)

    objects_box             = static_data['objects_box']
    location_coordinates    = static_data['location_coordinates']

    def euclidean_distance(x1,y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    while True:
        if not manipulation_action_type_sub._check_empty_data():
            if manipulation_action_type_sub._data == "Pick":
                rospy.loginfo("--> MANIPULATION PICK: waiting for desired object...")
                while True:
                    if not target_object_sub._check_empty_data():
                        rospy.loginfo("--> MANIPULATION PICK: performing arm moving action")      

                        t_picking = 5

                        rospy.sleep(t_picking)

                        xr,yr = position_subscriber.get_position_xy()
                        lc = location_coordinates[objects_box[target_object_sub._data]]
                        xb, yb = lc["x"], lc["y"]

                        distance = euclidean_distance(xr,yr, xb,yb)
                        print("@@@@ distance = ", distance)
                        if distance <= 0.6:
                            p = 1
                        else:
                            p = 0
                        
                        sample = np.random.rand()
                        
                        if sample <= p:
                            msg = True
                        else:
                            msg = False

                        msg = True

                        rospy.loginfo(f"picking action success = {msg}")
                        manipulation_done_pub._publish_msg(True)
                        manipulation_success_pub._publish_msg(msg)

                        manipulation_action_type_sub._reset_data()
                        target_object_sub._reset_data()
                    
                        break

            elif manipulation_action_type_sub._data == "Throw":

                rospy.loginfo("--> MANIPULATION THROW: waiting for desired object and tray...")
                while True:
                    if not target_object_sub._check_empty_data():
                        if not target_tray_sub._check_empty_data():
                            rospy.loginfo("--> MANIPULATION THROW: performing arm moving action")      
                            
                            t_throwing = 7

                            rospy.sleep(t_throwing)
                            xr,yr = position_subscriber.get_position_xy()
                            lc = location_coordinates[target_tray_sub._data]
                            xt, yt = lc["x"], lc["y"]

                            distance = euclidean_distance(xr,yr, xt,yt)

                            if distance <= 1:
                                p = 1
                            elif distance > 2:
                                p = 0
                            else:
                                p = 1 - (distance - 1)
                            
                            sample = np.random.rand()
                            
                            if sample <= p:
                                msg = True
                            else:
                                msg = False

                            msg = True
                            
                            rospy.loginfo(f"placing/throwing action success = {msg}")

                            manipulation_done_pub._publish_msg(True)
                            manipulation_success_pub._publish_msg(msg)

                            manipulation_action_type_sub._reset_data()
                            target_object_sub._reset_data()
                            target_tray_sub._reset_data()
                        
                            break
                        
            
            elif manipulation_action_type_sub._data == "Drop":
                rospy.loginfo("--> MANIPULATION DROP: dropping current object...")
                while True:
                    if not target_object_sub._check_empty_data():
                        rospy.loginfo("--> MANIPULATION DROP: performing arm moving action")      
                        
                        t_dropping = 1

                        rospy.sleep(t_dropping)

                        manipulation_done_pub._publish_msg(True)
                        manipulation_success_pub._publish_msg(True)

                        manipulation_action_type_sub._reset_data()
                        target_object_sub._reset_data()
                    
                        break
                        
            

        rospy.sleep(.3)
# %%
