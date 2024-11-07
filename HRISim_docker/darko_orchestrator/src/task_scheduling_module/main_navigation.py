#!/usr/bin/env python3

#%%
import numpy as np
import json
from orchestrator_class import Orchestrator
import tf
import math
import random
import time 
from std_msgs.msg import Int64MultiArray, MultiArrayDimension
import rospy
import pandas as pd


# load static data
path_to_static_data = "../static_data"

static_data_names = ["location_coordinates",
                        "action_graph_nodes",
                        "action_graph_conversion_dict",
                        "objects_box",
                        "rewards",
                        "action_times",
                        "items", "parameters"]

static_data = {}

for static_data_name in static_data_names:
    with open(path_to_static_data+"/"+static_data_name+".json", "r") as json_file:
        static_data[static_data_name] = json.load(json_file)

action_nodes            = list(static_data["action_graph_nodes"].keys())
trays                   = static_data['items']['trays']
objects                 = static_data['items']['objects']
rewards                 = static_data['rewards']
action_times            = static_data['action_times']
action_graph_nodes      = static_data['action_graph_nodes'] 
objects_box             = static_data['objects_box']
location_coordinates    = static_data['location_coordinates']
action_graph_nodes_int  = {static_data['action_graph_conversion_dict'][n]:static_data['action_graph_nodes'][n] for n in action_nodes }
action_nodes_int        = list(action_graph_nodes_int.keys())
parameters              = static_data["parameters"]


params = parameters
orchestrator_module = Orchestrator(action_graph_nodes_int, action_nodes_int, 
                                   "../static_data","../risk_awareness_module/manipulation_models",
                                    params,action_nodes,trays,objects,rewards,action_times,action_graph_nodes, objects_box,location_coordinates)


report = np.zeros((3,1,1), dtype=np.int64)
report_layout_dim = [MultiArrayDimension(label='x', size=report.shape[0], stride=0),MultiArrayDimension(label='y', size=report.shape[1], stride=0),MultiArrayDimension(label='z', size=report.shape[2], stride=0)]



msg = {"start":[], "end":[], "actual_time":[], "estimated_time":[]}
sublist_act_nodes = [4, 10, 14, 18]
for i in range(3):
    # get actual position
    rospy.sleep(0.2)

    p0,_ = orchestrator_module.get_closest_node(orchestrator_module.position_subscriber.get_position_xy())

    # get target position
    angle = np.random.uniform(low=-math.pi, high=math.pi, size=1)
    xo,yo,zo,wo = tf.transformations.quaternion_from_euler(0,0,angle[0])
    print(p0)
    if p0 not in sublist_act_nodes:
        p0 = 18
    sublist_act_nodes.remove(p0)
    p1 = random.choice(sublist_act_nodes)
    sublist_act_nodes.append(p0)
    xp= action_graph_nodes_int[p1]["x"]
    yp= action_graph_nodes_int[p1]["y"]

    nav_rsk_mtx, _, _ = orchestrator_module.retrieve_risk_mtx()
    t_min = nav_rsk_mtx[p0,p1,0]
    ts = nav_rsk_mtx[p0,p1,1]
    t_max = nav_rsk_mtx[p0,p1,2]
    r = nav_rsk_mtx[p0,p1,3]
    print(t_min, ts, t_max, r)
    # move to target position
    start_time = time.perf_counter()
    orchestrator_module.submit_goal(xp,yp,xo,yo,zo,wo)
    t = time.perf_counter() - start_time

    # save navigation data 
    msg["start"] += [p0]
    msg["end"] += [p1]
    msg["actual_time"] += [t]
    msg["estimated_time"] += [ts]

    report = Int64MultiArray()
    report.layout.dim = report_layout_dim
    report.data = np.array([int(p0), p1, int(t)]).flatten().tolist()

    # rospy.loginfo(report)
    orchestrator_module.send_navigation_report_pub._publish_msg(report)
    orchestrator_module.send_report_done_pub._publish_msg(True)


    # rospy.loginfo("pubblicato")

df = pd.DataFrame(msg)

# percorso_file_csv = 'times_1501_alpha01.csv'

# # Scrive il DataFrame su un file CSV
# df.to_csv(percorso_file_csv, index=False)



# %%
