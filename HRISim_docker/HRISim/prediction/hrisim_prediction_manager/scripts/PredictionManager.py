#!/usr/bin/env python


import math
import os
import pickle
import numpy as np
import pandas as pd
import rospy
import hrisim_util.ros_utils as ros_utils
import hrisim_util.constants as constants
from hrisim_prediction_srvs.srv import GetRiskMap, GetRiskMapResponse
from peopleflow_msgs.msg import WPPeopleCounters, Time as pT
from causalflow.basics.constants import *
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine
from collections import deque
import networkx as nx

           
class PredictionManager:
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        self.WPs = {}
        self.PDs = {}

        self.TOD = ''        
        self.hhmmss = ''
        self.elapsed = 0

        # subscribers
        rospy.Subscriber("/peopleflow/counter", WPPeopleCounters, self.cb_people_counter)
        rospy.Subscriber("/peopleflow/time", pT, self.cb_time)
        
        self.CIE = CausalInferenceEngine.load(CIEDIR)
        self.DAG = self.CIE.DAG['complete']
        self.MAX_LAG = self.DAG.max_lag
        
        self.calculation_order = list(nx.topological_sort(self.CIE.DAG2NX(self.DAG)))
            
        self.observations = deque(maxlen=self.MAX_LAG + 1) # Store up to MAX_LAG + 1 steps (current and previous)
        self.service = None
        
        rospy.set_param('/hrisim/prediction_ready', True)
          
    
    def cb_people_counter(self, wps: WPPeopleCounters):
        self.peopleAtWork = wps.numberOfWorkingPeople
        for wp in wps.counters:
            self.WPs[wp.WP_id.data] = wp.numberOfPeople
            self.PDs[wp.WP_id.data] = wp.numberOfPeople/WPS_COORD[wp.WP_id.data]['A']
            
            
    def cb_time(self, t: pT):
        self.TOD = int(ros_utils.seconds_to_hh(t.elapsed))
        self.hhmmss = t.hhmmss.data
        self.elapsed = t.elapsed
                       
            
    def collect_data(self):
        """
        Collects the current state of all data and logs or processes it.
        """
        # Check that self.PDs has exactly the same keys as WPS_COORD
        if set(self.PDs.keys()) != set(WPS_COORD.keys()):
            rospy.logerr("Mismatch between PDs keys and WPS_COORD keys")
            return
        
        # Create a dictionary for the current data
        current_data = {
            "TOD": self.TOD
        }
        for wp in WPS_COORD.keys():
            current_data[f"PD_{wp}"] = self.PDs[wp]

        # Add current data to the sliding window
        self.observations.append(current_data)
        if self.service is None: 
            self.service = rospy.Service('/get_risk_map', GetRiskMap, PM.handle_get_risk_map)
            
            
    def elapsed2TOD(self, t):
        d = 0
        for time in SCHEDULE:
            d += SCHEDULE[time]['duration']
            if t > d:
                continue
            else:
                return constants.TODS[SCHEDULE[time]['name']]
        
        
    def handle_get_risk_map(self, req):
        steps = [0, 40, 80, 119]
        treatment_len = 120
        # rospy.logwarn(f"Treatment length: {treatment_len}")
        # rospy.logwarn(f"Treatment seconds: {treatment_len*PREDICTION_STEP}")
        
        # Convert the observations deque to a pandas DataFrame
        data = pd.DataFrame(list(self.observations))
        
        # Init output
        flattened_PDs = []
        
        for i, wp in enumerate(SELECTED_WPS):
                       
            # For each waypoint, pass the corresponding data to the causal inference engine
            wp_obs = data[["TOD", f"PD_{wp}"]].values
            wp_obs_df = pd.DataFrame(wp_obs, columns=["TOD", "PD"])
            wp_obs_df["WP"] = constants.WPS[wp]
            
            # Init prior knowledge
            prior_knowledge = {'WP': np.full(treatment_len, wp_obs_df['WP'].values[-1])}
            
            # start_time_cie = time.time()
            res = self.CIE.whatIf('TOD', 
                                  [self.elapsed2TOD(self.elapsed + i * PREDICTION_STEP) for i in range(treatment_len)], 
                                  wp_obs_df.values,
                                  prior_knowledge,
                                  self.calculation_order
                                 )

            prediction_df = pd.DataFrame(res, columns=["TOD", "PD", "WP"])
            flattened_PDs.extend(np.nan_to_num(prediction_df['PD'].values[steps], nan=0.0))
        
        return GetRiskMapResponse(SELECTED_WPS, 
                                  len(steps), 
                                  len(SELECTED_WPS),
                                  flattened_PDs)
        

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('prediction_manager')
    
    CIEDIR = os.path.join(rospy.get_param("~CIE"), "cie.pkl")
    PREDICTION_STEP = rospy.get_param("~pred_step")
    ROBOT_MAX_VEL = float(ros_utils.wait_for_param("/move_base/TebLocalPlannerROS/max_vel_x"))
    g_path = ros_utils.wait_for_param("/peopleflow_pedsim_bridge/g_path")
    with open(g_path, 'rb') as f:
        G = pickle.load(f)

    SCHEDULE = ros_utils.wait_for_param("/peopleflow/schedule")
    WPS_COORD = ros_utils.wait_for_param("/peopleflow/wps")
    SELECTED_WPS = [constants.WP.WA_1_R.value, constants.WP.WA_2_R.value, constants.WP.WA_3_R.value, constants.WP.WA_3_CR.value, constants.WP.WA_4_R.value, constants.WP.WA_5_R.value,
                    constants.WP.WA_1_C.value, constants.WP.WA_2_C.value, constants.WP.WA_3_C.value, constants.WP.WA_4_C.value, constants.WP.WA_5_C.value,
                    constants.WP.WA_1_L.value, constants.WP.WA_2_L.value, constants.WP.WA_3_L.value, constants.WP.WA_3_CL.value, constants.WP.WA_4_L.value, constants.WP.WA_5_L.value,
                    constants.WP.TARGET_1.value, constants.WP.TARGET_2.value, constants.WP.TARGET_3.value, constants.WP.TARGET_4.value, constants.WP.TARGET_5.value, constants.WP.TARGET_6.value, constants.WP.TARGET_7.value]
    for wp in WPS_COORD:
        WPS_COORD[wp]['A'] = math.pi * WPS_COORD[wp]['r']**2
    
    PM = PredictionManager()
    
    rate = rospy.Rate(1 / PREDICTION_STEP)
        
    rospy.loginfo("Starting periodic data collection.")
    while not rospy.is_shutdown():
        PM.collect_data()
        rate.sleep()