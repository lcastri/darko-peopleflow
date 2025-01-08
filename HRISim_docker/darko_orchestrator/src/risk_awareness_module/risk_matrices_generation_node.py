#!/usr/bin/env python3
#%%
import rospy
from std_msgs.msg import Bool,Int64MultiArray,MultiArrayDimension, Float64MultiArray
from utils_module.topic_manager import SubscriberManager,PublisherManager
from utils_module.subscribers import OccupancyGridManager, ReportSubscriber
from risk_estimation_class import RiskEstimation
import math


if __name__ == '__main__':

    rospy.init_node('risk_matrices_generation', anonymous=True)

    orchestrator_started = rospy.get_param("/orchestrator_started")
    rospy.loginfo("Waiting for the orchestrator...")
    while not orchestrator_started:
        rospy.sleep(5)
        orchestrator_started = rospy.get_param("/orchestrator_started")


    ### <---------- publishers  ----------> ###
    risk_estimation_done_for_scheduler_pub = PublisherManager("/risk_estimation/estimation_done_for_scheduler", Bool)
    navigation_risk_for_scheduler_pub      = PublisherManager("/risk_estimation/navigation_risk_for_scheduler", Float64MultiArray)
    picking_risk_for_scheduler_pub         = PublisherManager("/risk_estimation/picking_risk_for_scheduler"   , Int64MultiArray)
    throwing_risk_for_scheduler_pub        = PublisherManager("/risk_estimation/throwing_risk_for_scheduler"  , Int64MultiArray)

    risk_estimation_done_for_monitoring_pub = PublisherManager("/risk_estimation/estimation_done_for_monitoring", Bool)
    navigation_risk_for_monitoring_pub      = PublisherManager("/risk_estimation/navigation_risk_for_monitoring", Float64MultiArray)
    picking_risk_for_monitoring_pub         = PublisherManager("/risk_estimation/picking_risk_for_monitoring"   , Int64MultiArray)
    throwing_risk_for_monitoring_pub        = PublisherManager("/risk_estimation/throwing_risk_for_monitoring"  , Int64MultiArray)


    ### <---------- subscribers ----------> ###
    risk_estimation_request_scheduler_sub = SubscriberManager("/risk_estimation/risk_estimation_request_scheduler",Bool,False)
    risk_estimation_request_monitoring_sub = SubscriberManager("/risk_estimation/risk_estimation_request_monitoring",Bool,False)

    navigation_report_sub = ReportSubscriber("/risk_estimation/send_navigation_report", Float64MultiArray)
    picking_report_sub = ReportSubscriber("/risk_estimation/send_picking_report",Float64MultiArray)
    placing_report_sub = ReportSubscriber("/risk_estimation/send_placing_report",Float64MultiArray)
    # send_report_done_sub = SubscriberManager("/risk_estimation/send_report_done", Bool, False)

    costmap_subscriber = OccupancyGridManager("/move_base/global_costmap/costmap", False)
    gridmap_subscriber = OccupancyGridManager("/map", False)
    risk_estimation_module = RiskEstimation(costmap_subscriber=costmap_subscriber,
                                            gridmap_subscriber=gridmap_subscriber,
                                            static_data_path="../static_data",
                                            manipulation_model_path="manipulation_models")
    
    navigation_risk_mtx,pick_risk_mtx,throw_risk_mtx = risk_estimation_module.get_risk_estimations()

    nav_risk_layout_dim = [MultiArrayDimension(label='x', size=navigation_risk_mtx.shape[0], stride=0),MultiArrayDimension(label='y', size=navigation_risk_mtx.shape[1], stride=0),MultiArrayDimension(label='z', size=navigation_risk_mtx.shape[2], stride=0)]
    pik_risk_layout_dim = [MultiArrayDimension(label='x', size=pick_risk_mtx.shape[0], stride=0)      ,MultiArrayDimension(label='y', size=pick_risk_mtx.shape[1], stride=0)]
    trw_risk_layout_dim = [MultiArrayDimension(label='x', size=throw_risk_mtx.shape[0], stride=0)     ,MultiArrayDimension(label='y', size=throw_risk_mtx.shape[1], stride=0)]

    print("ready")


    while True:
        # Aggiornamento continuo dei parametri
        if risk_estimation_module.update_parameters:
            # update velocity parameter for navigation activities using real data
            if not navigation_report_sub._check_empty_data():
                # new actual data
                report_data = navigation_report_sub._report_data
                node_from, node_to = int(report_data[0]), int(report_data[1])
                t_real = report_data[2]
                # predicted data
                nav_risk_mtx   = navigation_risk_mtx_sched
                t_prev = nav_risk_mtx[node_from, node_to, 0]
                # update velocity parameter
                risk_estimation_module.update_velocity(t_real, t_prev)

                # Scrivi la velocità aggiornata su un file esterno
                # with open('v_alpha01.txt', 'a') as file:
                #     file.write(f'{risk_estimation_module.v_max}\n')

                navigation_report_sub._reset_data()
            
            # update the rbf weights for manipulation activities using real data
            if not picking_report_sub._check_empty_data():
                # new actual data
                new_data  = picking_report_sub._report_data
                xf, yf = new_data[0], new_data[1] # coordinates of the picking node
                xt, yt = new_data[2], new_data[3] # coordinates of the box
                success = int(new_data[4]) # boolean 
                distance = math.sqrt((xt - xf) ** 2 + (yt - yf) ** 2)
                # update rbf weights
                risk_estimation_module.rbf_interp_picking.update_weights(distance, success)
                
                picking_report_sub._reset_data()

            if not placing_report_sub._check_empty_data():
                # update weights
                new_data  = placing_report_sub._report_data
                xf, yf = new_data[0], new_data[1] # coordinates of the throwing node
                xt, yt = new_data[2], new_data[3] # coordinates of the tray
                success = int(new_data[4]) # boolean 
                distance = math.sqrt((xt - xf) ** 2 + (yt - yf) ** 2)
                # update rbf weights
                risk_estimation_module.rbf_interp_throwing.update_weights(distance, success)
            
                placing_report_sub._reset_data()    

        # Controllo se lo Scheduler ha chiesto una calcolo delle matrici di rischio
        if not risk_estimation_request_scheduler_sub._check_empty_data():
            # calcolo nuove matrici di rischio
            navigation_risk_mtx_sched,pick_risk_mtx_sched,throw_risk_mtx_sched = risk_estimation_module.get_risk_estimations()
            # print("@@@ picking risk mtx", pick_risk_mtx)
            # print("@@@ navigation risk mtx", navigation_risk_mtx[0,:,:])
            
            # genero messaggio di nav risk
            nav_risk_message = Float64MultiArray()
            nav_risk_message.layout.dim = nav_risk_layout_dim
            nav_risk_message.data = navigation_risk_mtx_sched.flatten().tolist()

            # genero messaggio di pick risk
            pik_risk_message = Int64MultiArray()
            pik_risk_message.layout.dim = pik_risk_layout_dim
            pik_risk_message.data = pick_risk_mtx_sched.flatten().tolist()

            # genero messaggio di throw risk
            trw_risk_message = Int64MultiArray()
            trw_risk_message.layout.dim = trw_risk_layout_dim
            trw_risk_message.data = throw_risk_mtx_sched.flatten().tolist()

            # invio i messaggi di nav, pick, throw risk
            navigation_risk_for_scheduler_pub._publish_msg(nav_risk_message)
            picking_risk_for_scheduler_pub._publish_msg(pik_risk_message)
            throwing_risk_for_scheduler_pub._publish_msg(trw_risk_message)

            # segnalo che ho inviato tutti i messaggi di rischio
            risk_estimation_done_for_scheduler_pub._publish_msg(True)

            # metto a False il flag di richiesta invio matrici di rischio
            risk_estimation_request_scheduler_sub._reset_data()

        # Controllo se il Monitoring ha chiesto una calcolo delle matrici di rischio
        if not risk_estimation_request_monitoring_sub._check_empty_data():

            # calcolo nuove matrici di rischio
            navigation_risk_mtx,pick_risk_mtx,throw_risk_mtx = risk_estimation_module.get_risk_estimations()
            # genero messaggio di nav risk
            nav_risk_message = Float64MultiArray()
            nav_risk_message.layout.dim = nav_risk_layout_dim
            nav_risk_message.data = navigation_risk_mtx.flatten().tolist()

            # genero messaggio di pick risk
            pik_risk_message = Int64MultiArray()
            pik_risk_message.layout.dim = pik_risk_layout_dim
            pik_risk_message.data = pick_risk_mtx.flatten().tolist()

            # genero messaggio di throw risk
            trw_risk_message = Int64MultiArray()
            trw_risk_message.layout.dim = trw_risk_layout_dim
            trw_risk_message.data = throw_risk_mtx.flatten().tolist()

            # invio i messaggi di nav, pick, throw risk
            navigation_risk_for_monitoring_pub._publish_msg(nav_risk_message)

            picking_risk_for_monitoring_pub._publish_msg(pik_risk_message)
            # print("throw risk mtx ", throw_risk_mtx_sched)
            throwing_risk_for_monitoring_pub._publish_msg(trw_risk_message)

            # segnalo che ho inviato tutti i messaggi di rischio
            risk_estimation_done_for_monitoring_pub._publish_msg(True)

            # metto a False il flag di richiesta invio matrici di rischio
            risk_estimation_request_monitoring_sub._reset_data()
        
        rospy.sleep(.2)



# %%
