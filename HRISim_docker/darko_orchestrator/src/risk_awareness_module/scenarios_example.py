#!/usr/bin/env python3
# %%
import rospy
from std_msgs.msg import Bool, Float64MultiArray
from topic_manager import SubscriberManager,PublisherManager
from subscribers import RiskMtxSubscriber, ScenariosSubscriber
from risk_monitoring_class import RiskMonitoring
from darko_orchestrator.msg import ScenarioList, Scenario, State

rospy.init_node('temp_node', anonymous=True)

mission_active_sub       = SubscriberManager("/risk_monitoring/mission_active_flag", Bool, False)
scenarios_sub            = ScenariosSubscriber("/risk_monitoring/scenarios")


# %%

scenario_lst = []
data = scenarios_sub._scenarios_data
n_scenarios = len(data.scenario_list)
for i in range(n_scenarios):
    scenario = data.scenario_list[i]
    state_list = scenario.state_list
    n_states = len(state_list)
    state_lst = []
    for j in range(n_states):
        state = state_list[j]
        state_lst.append(state.state)
    scenario_lst.append(state_lst)


# %%
