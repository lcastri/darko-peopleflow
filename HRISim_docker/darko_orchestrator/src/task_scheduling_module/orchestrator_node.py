#!/usr/bin/env python3

from utils_module.subscribers import OccupancyGridManager
import numpy as np
import json
from orchestrator_class import Orchestrator
import rospy
from large_graph_generation.main import generate_large_graph

rospy.init_node('orchestrator_node', anonymous=True)

costmap_subscriber = OccupancyGridManager("/move_base/global_costmap/costmap",True)

rospy.loginfo("generating graph...")
generate_large_graph(costmap_subscriber)
rospy.loginfo("graph generated, all other modules can start")
rospy.set_param("/orchestrator_started", True)

# load static data
path_to_static_data = "../static_data"

static_data_names = ["location_coordinates",
                        "action_graph_nodes",
                        "objects_box",
                        "rewards",
                        "action_times",
                        "items", "parameters",
                        "reduced_global_map_parameters"]

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
reduced_global_map_parameters = static_data['reduced_global_map_parameters']

action_graph_nodes_int  = {int(n[1:]):static_data['action_graph_nodes'][n] for n in action_nodes }
action_nodes_int        = list(action_graph_nodes_int.keys())

parameters              = static_data["parameters"]

params = parameters

orchestrator_module = Orchestrator(
    action_graph_nodes_int,
    action_nodes_int, 
    "../static_data",
    "../risk_awareness_module/manipulation_models",
    params,
    action_nodes,
    trays,
    objects,
    rewards,
    action_times,
    action_graph_nodes,
    objects_box,location_coordinates,
    costmap_subscriber,
    reduced_global_map_parameters
)

while True:
    rospy.loginfo("Waiting for mission...")
    t, final_state = orchestrator_module.wait_for_mission()
    rospy.loginfo(f"Mission completed in {t} with final state {final_state}")
    
# orchestrator_module.get_max_neighborhood_value(2.0, -1.4)

# missione = {"tray0":{"object0":1,"object1":0,"object2":1,"object3":0,"object4":1},
#             "tray1":{"object0":0,"object1":1,"object2":0,"object3":1,"object4":0}}

# t, mission_state= orchestrator_module.solve_mission(missione)
# print(t)
# print(mission_state)
# %%
