from utils_module.topic_manager import PublisherManager, SubscriberManager
from std_msgs.msg import Int64MultiArray, Float64MultiArray, Bool
from darko_orchestrator.msg import CurrentAction, State, Heatmap
import rospy

rospy.init_node('darko_web_ui', anonymous=True)

mission_pub             = PublisherManager('/web_ui/mission', Int64MultiArray)

current_action_sub      = SubscriberManager('/web_ui/current_action', CurrentAction, False)
current_state_sub       = SubscriberManager('/web_ui/current_state', State, False)
qfa_sub                 = SubscriberManager('/web_ui/qfa', State, False)
monitoring_risk_sub     = SubscriberManager("/web_ui/monitoring_risk", Float64MultiArray)

reschedule_sub          = SubscriberManager("/web_ui/reschedule", Bool, False)
heatmap_subscriber      = SubscriberManager("/web_ui/costmap", Heatmap, False)