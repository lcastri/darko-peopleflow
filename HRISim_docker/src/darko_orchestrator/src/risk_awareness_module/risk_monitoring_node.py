#!/usr/bin/env python3
#%%
import rospy
from std_msgs.msg import Bool, Float64MultiArray, Int64MultiArray
from std_srvs.srv import Empty
from utils_module.topic_manager import SubscriberManager,PublisherManager
from utils_module.subscribers import RiskMtxSubscriber, ScenariosSubscriber, RiskMtxSubscriberFloat
from risk_monitoring_class import RiskMonitoring
from darko_orchestrator.msg import ScenarioList, Scenario, State
from darko_orchestrator.msg import ScenarioList, Scenario, State

rospy.init_node('mimic_request', anonymous=True)

### <---------- publishers  ----------> ###
risk_estimation_request_pub = PublisherManager("/risk_estimation/risk_estimation_request_monitoring", Bool)
reschedule_trigger_pub = PublisherManager("/risk_monitoring/reschedule", Bool)
ui_reschedule_trigger_pub = PublisherManager("/web_ui/reschedule", Bool)
monitoring_risk_ui_pub = PublisherManager("/web_ui/monitoring_risk", Float64MultiArray)

### <---------- subscribers ----------> ###
mission_active_sub       = SubscriberManager("/risk_monitoring/mission_active_flag", Bool, False)

risk_estimation_done_sub = SubscriberManager("/risk_estimation/estimation_done_for_monitoring", Bool,False)
navigation_risk_sub      = RiskMtxSubscriberFloat("/risk_estimation/navigation_risk_for_monitoring")
picking_risk_sub         = RiskMtxSubscriber("/risk_estimation/picking_risk_for_monitoring"   )
throwing_risk_sub        = RiskMtxSubscriber("/risk_estimation/throwing_risk_for_monitoring"  )

navigation_risk_sched_sub = RiskMtxSubscriberFloat("/risk_estimation/navigation_risk_for_scheduler")
picking_risk_sched_sub    = RiskMtxSubscriber("/risk_estimation/picking_risk_for_scheduler"   )
throwing_risk_sched_sub   = RiskMtxSubscriber("/risk_estimation/throwing_risk_for_scheduler"  )

scenarios_sub                  = ScenariosSubscriber("/risk_monitoring/scenarios")
scenario_computation_done_sub  = SubscriberManager("/risk_monitoring/scenario_computation_done", Bool,False)

risk_monitoring_module = RiskMonitoring()


while True:

    # controllo se la missione è attiva
    # if not mission_active_sub._check_empty_data():

    #     rospy.wait_for_service('/move_base/clear_costmaps')
    #     try:
    #         clear_costmaps_service = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
    #         clear_costmaps_service()
    #         rospy.loginfo("Costmap cleared successfully!")
    #     except rospy.ServiceException as e:
    #         rospy.logerr("Service call failed: %s" % e)

    #     reschedule_trigger_pub._publish_msg(True)
    #     ui_reschedule_trigger_pub._publish_msg(True)
    #     print("triggerato")
       
        # if not scenario_computation_done_sub._check_empty_data():
        #     rospy.sleep(0.1)
        #     # prendo gli scenari (e le probabilità) generati dallo scheduler
        #     risk_monitoring_module.scenario_lst = scenarios_sub._scenarios_data
        #     risk_monitoring_module.probability_lst = scenarios_sub._probabilities_data
        
        #     # prendo le mappe di rischio calcolate per lo scheduler
        #     risk_monitoring_module.nav_risk_mtx   = navigation_risk_sched_sub._risk_data
        #     risk_monitoring_module.pick_risk_mtx  = picking_risk_sched_sub._risk_data
        #     risk_monitoring_module.throw_risk_mtx = throwing_risk_sched_sub._risk_data
            
        #     # scenarios_sub._reset_data()
        #     scenario_computation_done_sub._reset_data()
            
        #     old_trigger = False
            
        #     while scenario_computation_done_sub._check_empty_data():
                
        #         # richiesta nuovo calcolo rischi
        #         risk_estimation_request_pub._publish_msg(True)
        #         # print("richiesta fatta")

        #         if not risk_estimation_done_sub._check_empty_data():
        #             rospy.sleep(0.1)
        #             # prendo le mappe di rischio calcolate per il monitoring
        #             # print("stima fatta")
        #             risk_monitoring_module.updated_nav_risk_mtx   = navigation_risk_sub._risk_data
        #             risk_monitoring_module.updated_pick_risk_mtx  = picking_risk_sub._risk_data
        #             risk_monitoring_module.updated_throw_risk_mtx = throwing_risk_sub._risk_data

        #             risk_estimation_done_sub._reset_data()

        #             # valuto se è cambiato il rischio e se devo triggerare la rischedulazione
        #             trigger, nav_old_tot, nav_new_tot, manip_old_tot, manip_new_tot = risk_monitoring_module.reschedule_evaluation() # booleano
        #             msg = Float64MultiArray()
        #             msg.data = [nav_old_tot, nav_new_tot, manip_old_tot, manip_new_tot]
        #             monitoring_risk_ui_pub._publish_msg(msg)
        #             if trigger and old_trigger:
        #                 reschedule_trigger_pub._publish_msg(True)
        #                 ui_reschedule_trigger_pub._publish_msg(True)
        #                 print("triggerato")
        #                 break

        #             old_trigger = trigger
        #         rospy.sleep(1)



    rospy.sleep(60)


        

# %%
