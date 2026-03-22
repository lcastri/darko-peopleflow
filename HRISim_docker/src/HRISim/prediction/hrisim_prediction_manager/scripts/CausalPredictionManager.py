#!/usr/bin/env python

import math
import pickle
import numpy as np
import rospy
import hrisim_util.ros_utils as ros_utils
import hrisim_util.constants as constants
from hrisim_prediction_srvs.srv import GetRiskMap, GetRiskMapResponse
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from peopleflow_msgs.msg import WPPeopleCounters, Time as pT
from robot_msgs.msg import BatteryStatus
from std_msgs.msg import String
import networkx as nx
import pyAgrum
import pyAgrum.causal as pyc
from enum import Enum
import time


class Robot():
    def __init__(self) -> None:
        self.x = None
        self.y = None
        self.yaw = 0
        self.v = 0
        self.battery_level = 0
        self.is_charging = 0
        self.closest_wp = ''
        self.task_result = 0
                

def heuristic(a, b):
    pos = nx.get_node_attributes(G, 'pos')
    (x1, y1) = pos[a]
    (x2, y2) = pos[b]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_info(D, auditDict, var):
    if auditDict[var]['method'] == 'NoDiscretization':
        edges = auditDict[var]['values']
        midpoints = auditDict[var]['values']
        return 'NoDiscretization', edges, midpoints
    elif 'param' in auditDict[var] and isinstance(auditDict[var]['param'], list):
        edges = auditDict[var]['param']
        midpoints = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(edges)-1)]
        return auditDict[var]['param'], edges, midpoints
    quantiles = np.linspace(0, 100, auditDict[var]['param'] + 1 if 'param' in auditDict[var] else auditDict[var]['nbBins'] + 1)
    edges = np.percentile(D[var].values, quantiles)
    midpoints = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(edges)-1)]
    return quantiles, edges, midpoints
       

class PredictionManager:
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        self.robot = Robot()

        self.Ws = {}
        self.Ds = {}

        self.S = ''        
        self.hhmmss = ''
        self.elapsed = 0

        # subscribers
        rospy.Subscriber("/robot_pose", PoseWithCovarianceStamped, self.cb_robot_pose)
        rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.cb_odom)
        rospy.Subscriber("/peopleflow/counter", WPPeopleCounters, self.cb_people_counter)
        rospy.Subscriber("/peopleflow/time", pT, self.cb_time)
        rospy.Subscriber("/hrisim/robot_battery", BatteryStatus, self.cb_robot_battery)
        rospy.Subscriber("/hrisim/robot_closest_wp", String, self.cb_robot_closest_wp)
        
        AUDITs = pickle.load(open(f"{CIEDIR}/AUDITs.pkl", "rb"))
        Ds = pickle.load(open(f"{CIEDIR}/Ds.pkl", "rb"))
        self.CIE = {}
        for wp in WPS_COORD.keys():
            if wp in ['parking', 'charging-station']: continue
            bn = pyAgrum.loadBN(f"{CIEDIR}/CIE_{wp}.bifxml")
            cm = pyc.CausalModel(bn)
            self.CIE[wp] = {'audit': AUDITs[wp], 'd': Ds[wp], 'bn': bn, 'cm': cm}
            
        quantiles_V, edges_V, midpoints_V = get_info(Ds['r1'], AUDITs['r1'], "V") # r1 has been chosen as random wp
        quantiles_L, edges_L, midpoints_L = get_info(Ds['r1'], AUDITs['r1'], "L") # r1 has been chosen as random wp
        self.V_info = {'quantiles': quantiles_V, 'edges': edges_V, 'midpoints': midpoints_V}
        self.L_info = {'quantiles': quantiles_L, 'edges': edges_L, 'midpoints': midpoints_L}
               
        self.TTWP = self.get_ttwp()
        self.ARCs = []
        for arc in self.TTWP.keys():
            self.ARCs.append('__'.join(arc))
        rospy.Service('/hrisim/riskMap/predict', GetRiskMap, self.handle_get_risk_map)
        
        rospy.set_param('/hrisim/prediction_ready', True)
        rospy.logwarn("Prediction Manager ready!")     
                   
                                      
    def cb_robot_pose(self, pose: PoseWithCovarianceStamped):
        self.robot.x, self.robot.y, self.robot.yaw = ros_utils.getPose(pose.pose.pose)
        
        
    def cb_odom(self, odom: Odometry):
        self.robot.v = abs(odom.twist.twist.linear.x)
        
        
    def cb_robot_closest_wp(self, wp: String):
        self.robot.closest_wp = wp.data
        
        
    def cb_robot_battery(self, b: BatteryStatus):
        self.robot.battery_level = b.level.data
        self.robot.is_charging = b.is_charging.data
    
    
    def cb_people_counter(self, wps: WPPeopleCounters):
        self.peopleAtWork = wps.numberOfWorkingPeople
        for wp in wps.counters:
            self.Ws[wp.WP_id.data] = wp.numberOfPeople
            self.Ds[wp.WP_id.data] = wp.numberOfPeople/WPS_COORD[wp.WP_id.data]['A']
            
            
    def cb_time(self, t: pT):
        self.S = int(ros_utils.seconds_to_hh(t.elapsed))
        self.hhmmss = t.hhmmss.data
        self.elapsed = t.elapsed
            
            
    def get_ttwp(self, relative=False):
        """
        Calculate the time to travel between each pair of waypoints in the graph.

        Returns:
            dict: A dictionary with waypoint pairs as keys and travel times as values.
        """
        travelled_time = {}
        if relative:
            for wp in WPS_COORD.keys():
                path = nx.astar_path(G, self.robot.closest_wp, wp, heuristic=heuristic, weight='weight')
                travelled_distance = 0
                for wp_idx in range(1, len(path)):
                    wp_current = path[wp_idx-1]
                    wp_next = path[wp_idx]
                    travelled_distance += math.sqrt((WPS_COORD[wp_next]['x'] - WPS_COORD[wp_current]['x'])**2 + (WPS_COORD[wp_next]['y'] - WPS_COORD[wp_current]['y'])**2)
                travelled_time[wp] = math.ceil((travelled_distance/ROBOT_MAX_VEL)/PREDICTION_STEP)
        else:
            for arc in G.edges():
                wp_i, wp_j = arc
                travelled_distance = math.sqrt((WPS_COORD[wp_i]['x'] - WPS_COORD[wp_j]['x'])**2 + (WPS_COORD[wp_i]['y'] - WPS_COORD[wp_j]['y'])**2)
                travelled_time[arc] = math.ceil((travelled_distance/ROBOT_MAX_VEL)/PREDICTION_STEP)
        return travelled_time
                       
            
    def elapsed2TOD(self, t):
        d = 0
        for time in SCHEDULE:
            d += SCHEDULE[time]['duration']
            if t > d:
                continue
            else:
                return constants.TODS[SCHEDULE[time]['name']] - 1 # -1 because the CIE doesnt consider STARTING
            
            
    def predict_L(self, v, tod, wp):
        
        def find_bin(value, edges):
            """
            Given a continuous value and an array of bin edges,
            return the index of the bin that contains the value.
            """
            idx = np.digitize(value, edges, right=False) - 1
            return int(max(0, min(idx, len(edges) - 2)))
    
        RV_bin_idx = find_bin(v, self.V_info['edges'])
        tod_bin = tod
        wp_bin = 0
                                        
        # --- CausalModel prediction ---
        cm = self.CIE[wp]['cm']
        evidence = {"V": RV_bin_idx, "S": tod_bin, "W": wp_bin}
        _, adj, _ = pyc.causalImpact(cm, on="L", doing="V", knowing={"W", "S"}, values=evidence)
        posterior_causal = adj.toarray()
        pred_causal = sum(posterior_causal[j] * self.L_info['midpoints'][j] for j in range(len(posterior_causal)))
                        
        return pred_causal
    
    
    def predict_D(self, tod, wp):

        wp_bin = 0
        tod_bin = tod
              
        cm = self.CIE[wp]['cm']
        dwp = self.CIE[wp]['d']
               
        _, _, midpoints_PD = get_info(dwp, self.CIE[wp]['audit'], 'D')
      
        # --- CausalModel prediction ---
        evidence = {"S": tod_bin, "W": wp_bin}
        _, adj, _ = pyc.causalImpact(cm, on="D", doing="S", knowing={"W"}, values=evidence)
        posterior_causal = adj.toarray()
        pred_causal = sum(posterior_causal[j] * midpoints_PD[j] for j in range(len(posterior_causal)))

        return pred_causal
    

    def handle_get_risk_map(self, req):
        start_time = time.perf_counter()
        rospy.logwarn("################# Prediction requested #################")
        
        # Convert the observations deque to a pandas DataFrame
        TTWP_relative = self.get_ttwp(relative=True)
               
        # Init output
        D_wps = {}
        D_infs = {}
        Ds = []
        Ls = []
        D_inf_time = []
        L_inf_time = []
        
        tod = self.elapsed2TOD(self.elapsed)
        rospy.logwarn(f"    TOD: {(list(constants.TODS.keys())[tod + 1])}")
        
        for wp in WPS_COORD.keys():
            traversal_step = TTWP_relative[wp]
            tod = self.elapsed2TOD(self.elapsed + traversal_step * PREDICTION_STEP)
            D_start_time = time.perf_counter()
            D_wps[wp] = self.predict_D(tod, wp)
            D_end_time = time.perf_counter()
            D_infs[wp] = D_end_time - D_start_time
            
        for arc in self.ARCs:
            wp_i, wp_j = arc.split('__')
            traversal_step = self.TTWP[(wp_i, wp_j)]
                        
            L_start_time = time.perf_counter()
            Ls.append(self.predict_L(ROBOT_MAX_VEL, tod, self.robot.closest_wp)*traversal_step)
            L_end_time = time.perf_counter()
            L_inf_time.append(L_end_time - L_start_time)
            Ds.append((D_wps[wp_i] + D_wps[wp_j])/2)
            D_inf_time.append(D_infs[wp_i] + D_infs[wp_j])
            
        end_time = time.perf_counter()
        tot_inf_time = end_time - start_time
            
        return GetRiskMapResponse(self.ARCs, Ds, Ls, tot_inf_time, D_inf_time, L_inf_time)
        

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('prediction_manager')
    
    CIEDIR = rospy.get_param("~CIE")
    PREDICTION_STEP = rospy.get_param("~pred_step")
    ROBOT_MAX_VEL = float(ros_utils.wait_for_param("/move_base/TebLocalPlannerROS/max_vel_x"))
    g_path = ros_utils.wait_for_param("/peopleflow_pedsim_bridge/g_path")
    with open(g_path, 'rb') as f:
        G = pickle.load(f)

    SCHEDULE = ros_utils.wait_for_param("/peopleflow/schedule")
    WPS_COORD = ros_utils.wait_for_param("/peopleflow/wps")
    for wp in WPS_COORD:
        WPS_COORD[wp]['A'] = math.pi * WPS_COORD[wp]['r']**2
    
    PM = PredictionManager()
    
    rate = rospy.Rate(1 / PREDICTION_STEP)
        
    while not rospy.is_shutdown():
        rate.sleep()