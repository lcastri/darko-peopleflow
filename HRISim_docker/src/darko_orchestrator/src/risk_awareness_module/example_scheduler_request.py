#!/usr/bin/env python3
#%%
import rospy
from std_msgs.msg import Bool
from topic_manager import SubscriberManager,PublisherManager
from subscribers import RiskMtxSubscriber
import json
import pathlib

rospy.init_node('mimic_request', anonymous=True)

### <---------- publishers  ----------> ###
risk_estimation_request_pub = PublisherManager("/risk_estimation/risk_estimation_request_scheduler", Bool)
mission_active_pub         = PublisherManager("/risk_monitoring/mission_active_flag", Bool)

### <---------- subscribers ----------> ###
risk_estimation_done_sub = SubscriberManager("/risk_estimation/estimation_done_for_scheduler", Bool,False)
navigation_risk_sub      = RiskMtxSubscriber("/risk_estimation/navigation_risk_for_scheduler")
picking_risk_sub         = RiskMtxSubscriber("/risk_estimation/picking_risk_for_scheduler"   )
throwing_risk_sub        = RiskMtxSubscriber("/risk_estimation/throwing_risk_for_scheduler"  )

#%%

# inizio missione
mission_active_pub._publish_msg(True)

# richiesta calcolo rischi
risk_estimation_request_pub._publish_msg(True)

while True:
    if not risk_estimation_done_sub._check_empty_data():

        nav_risk_mtx   = navigation_risk_sub._risk_data
        pick_risk_mtx  = picking_risk_sub._risk_data
        throw_risk_mtx = throwing_risk_sub._risk_data
        
        print("done")
        risk_estimation_done_sub._reset_data()
        break


# %%
