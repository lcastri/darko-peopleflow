import rospy
import numpy as np
import time
from std_msgs.msg import Bool,String,Int64MultiArray,Float64MultiArray
from darko_orchestrator.msg import ScenarioList, Scenario, State, CurrentAction, Action
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import tf
import math
import sys
from subscribers import AmclPoseManager, OccupancyGridManager,RiskMtxSubscriber, RiskMtxSubscriberFloat
from scheduler_class import Scheduler
from topic_manager import SubscriberManager,PublisherManager
from utils import global_costamap_reduction

class Orchestrator:

    def __init__(self, action_graph_nodes_int, action_nodes_int, static_data_path,manipulation_model_path,
                params,action_nodes,trays,objects,rewards,action_times,action_graph_nodes, objects_box,location_coordinates, costmap_subscriber, reduced_global_map_parameters):
        
        self.action_graph_nodes_int = action_graph_nodes_int
        self.action_nodes_int       = action_nodes_int
        self.action_graph_nodes     = action_graph_nodes
        self.action_nodes           = action_nodes
        self.params                 = params
        self.trays                  = trays
        self.objects                = objects
        self.rewards                = rewards
        self.action_times           = action_times
        self.objects_box            = objects_box
        self.location_coordinates   = location_coordinates
        self.reduced_global_map_parameters = reduced_global_map_parameters

        ### <---------- publishers  ----------> ###
        self.manipulation_action_type_pub   = PublisherManager("/manipulation/action_type",String)
        self.target_object_pub              = PublisherManager("/manipulation/object_type",String)
        self.target_tray_pub                = PublisherManager("/manipulation/tray_type"  ,String)
        self.risk_estimation_request_pub    = PublisherManager("/risk_estimation/risk_estimation_request_scheduler", Bool)
        self.mission_active_pub             = PublisherManager("/risk_monitoring/mission_active_flag", Bool)
        self.scenarios_pub                  = PublisherManager("/risk_monitoring/scenarios", ScenarioList)
        self.scenario_computation_done_pub  = PublisherManager("/risk_monitoring/scenario_computation_done", Bool)
        self.send_navigation_report_pub     = PublisherManager("/risk_estimation/send_navigation_report", Float64MultiArray)
        self.send_picking_report_pub        = PublisherManager("/risk_estimation/send_picking_report", Float64MultiArray)
        self.send_placing_report_pub        = PublisherManager("/risk_estimation/send_placing_report", Float64MultiArray)
        self.send_report_done_pub           = PublisherManager("/risk_estimation/send_report_done", Bool)
        self.ui_current_action_pub          = PublisherManager("/web_ui/current_action", CurrentAction)
        self.ui_current_state_pub           = PublisherManager("/web_ui/current_state", State)
        self.ui_qfa_pub                     = PublisherManager("/web_ui/qfa", State)
        
        ### <---------- subscribers ----------> ###
        self.manipulation_done_sub    = SubscriberManager("/manipulation/action_finished", Bool, False)
        self.manipulation_success_sub = SubscriberManager("/manipulation/action_success" , Bool, False)
        self.reschedule_sub           = SubscriberManager("/risk_monitoring/reschedule", Bool, False)
        self.risk_estimation_done_sub = SubscriberManager("/risk_estimation/estimation_done_for_scheduler", Bool,False)
        self.navigation_risk_sub      = RiskMtxSubscriberFloat("/risk_estimation/navigation_risk_for_scheduler")
        self.picking_risk_sub         = RiskMtxSubscriber("/risk_estimation/picking_risk_for_scheduler"   )
        self.throwing_risk_sub        = RiskMtxSubscriber("/risk_estimation/throwing_risk_for_scheduler"  )
        self.ui_mission_subscriber = SubscriberManager("/web_ui/mission", Int64MultiArray, False)

        self.costmap_subscriber = costmap_subscriber

        self.safety_threshold = 40
        self.neighborhood_size = 5
        self.scheduler_module = Scheduler(params,action_nodes,trays,objects,rewards,action_times)

        self.client = actionlib.SimpleActionClient('/move_base',MoveBaseAction)
        self.client.wait_for_server()
        self.position_subscriber = AmclPoseManager()
        
        self.reduced_map = global_costamap_reduction(self.costmap_subscriber, self.reduced_global_map_parameters)

        self.max_mission_steps = params["max_mission_steps"]
        self.go_to_closest_node()


    def wait_for_mission(self):

        while self.ui_mission_subscriber._check_empty_data():
            rospy.sleep(1)
        data = self.ui_mission_subscriber._data
        self.ui_mission_subscriber._reset_data()

        mission = self.mission_to_dict(data)
        rospy.loginfo(f"received mission {mission}")

        return self.solve_mission(mission)


    def mission_to_dict(self, data):

        mission = {}

        for t in range(len(self.trays)):
            mission[self.trays[t]] = {}
            for o in range (len(self.objects)):
                mission[self.trays[t]][self.objects[o]] = data[o + t * len(self.objects)]

        return mission


    def go_to_closest_node(self, with_safety=False):
        closest_node, closest_distance = self.get_closest_node(self.position_subscriber.get_position_xy(), with_safety=with_safety)
        if closest_distance>0.:
            xp,yp = self.action_graph_nodes_int[closest_node]['x'],self.action_graph_nodes_int[closest_node]['y']
            xo,yo,zo,wo = 0,0,0,1
            self.submit_goal(xp,yp,xo,yo,zo,wo)
        return closest_node
        
        #testare cancel goal di action lib inserendo wait_for_result nel while (chiamate continue con duration molto piccola)
        #se arrivano prima i risultati, bene, altrimenti se arriva il comando stop_robot allora cancello il goal e stoppo il
        #robot come fatto in test_robot_stop (verificare)
    
    def stop_robot(self):
        goal = MoveBaseGoal()
        # Specifica qui la posizione corrente del robot o l'obiettivo da interrompere
        goal.target_pose.header.frame_id = 'base_link'
        goal.target_pose.pose.position.x = 0.0
        goal.target_pose.pose.position.y = 0.0
        goal.target_pose.pose.orientation.w = 1.0
        # Invia il goal per interrompere l'azione corrente
        self.client.send_goal(goal)
        self.client.wait_for_result()
    def get_corrected_cost(self, cost):

        if cost < 50:
            return 0
        if  cost < 80:
            return cost/3
        return 100    


    def get_max_neighborhood_value(self, x, y):

        # crea un quadrato intorno al punto x,y di dimensione neighborhood_size e trova il valore max della costmap in quella zona
        col_index, row_index = self.reduced_map.get_costmap_x_y(x,y)
        i_min = max(0, row_index - self.neighborhood_size)
        i_max = min(row_index + self.neighborhood_size, self.reduced_map.height -1)
        j_min = max(0, col_index - self.neighborhood_size)
        j_max = min(col_index + self.neighborhood_size, self.reduced_map.width -1)

        # self.reduced_map.plot_map_points(j_min, i_min, self.neighborhood_size, self.action_graph_nodes, i_min, i_max, j_min, j_max, col_index , row_index)
        values = [
            self.get_corrected_cost(self.reduced_map.get_cost_from_costmap_x_y(j,i))
            for i in range(i_min, i_max +1 ) for j in range(j_min, j_max +1)
        ]
        return max(values)



    def get_closest_node(self,xy_robot, with_safety=False):

        self.reduced_map.update()
        
        x_robot,y_robot = xy_robot
        closest_node,closest_distance = self.action_nodes_int[0],10**4
        for n in self.action_nodes_int:
            x = self.action_graph_nodes_int[n]['x']
            y = self.action_graph_nodes_int[n]['y']

            dist = ((x-x_robot)**2 + (y-y_robot)**2 )**.5

            if with_safety:
                cost = self.get_max_neighborhood_value(x,y)
                if cost > self.safety_threshold:
                    continue
            if dist < closest_distance:
                closest_node = n
                closest_distance = dist

        return closest_node,closest_distance
    
    
    def check_for_reschedule(self):
        if not self.reschedule_sub._check_empty_data():
            rospy.loginfo("reschedule triggered")
            self.reschedule_sub._reset_data()
            self.stop_robot()
            return True
        return False
    '''
    per lo splitting delle missioni:

    inizialmente (prima del while) richiamare il retrieve risk matrices e ordinare gli step delle missioni
    considerando nav(current, box) + pick + nav(box, tray) + place/throw 

    scegliere le prime X missioni (5 oggetti di tipo diverso o 3 tipi diversi di oggetti con molteplicità 2)

    nel while, ad ogni return del solve_mission_internal (o perché ha finito, o perché ha terminato l'horizon o a causa di un rescheduling)
    dobbiamo capire se ha cose a bordo: se le ha teniamo gli step di placing e "rabbocchiamo" la missione estraendo le n rimanenti
    considerando tempi e rischi aggiornati

    il dropping lo gestisce lo scheduler (ancora da tunare)

    '''

    def estimate_step_time_and_risk(self, object_idx, tray_idx,nav_risk_mtx,picking_prob_mtx,throwing_prob_mtx):

        robot_node, _ = self.get_closest_node(self.position_subscriber.get_position_xy())

        box_coords = self.location_coordinates[self.objects_box[self.objects[object_idx]]]
        tray_coords = self.location_coordinates[self.trays[tray_idx]]

        box_node, _ = self.get_closest_node((box_coords["x"], box_coords["y"]))
        tray_node, _ = self.get_closest_node((tray_coords["x"], tray_coords["y"]))

        time = 0
        risk = 0

        ## nav to box
        time += nav_risk_mtx[robot_node][box_node][1]
        risk += nav_risk_mtx[robot_node][box_node][3]

        ## pick
        time += self.scheduler_module.picking_time
        risk += 100 - picking_prob_mtx[box_node][object_idx]

        ## nav to tray
        time += nav_risk_mtx[box_node][tray_node][1]
        risk += nav_risk_mtx[box_node][tray_node][3]

        ## place
        time += self.scheduler_module.throwing_time
        risk += 100 - throwing_prob_mtx[tray_node][tray_idx]

        return time, risk
        

    def generate_mission_steps(self, missione):

        step_list = []
        for t in range(self.scheduler_module.n_trays):
            for o in range(self.scheduler_module.n_objects):
                n = missione[self.trays[t]][self.objects[o]]
                for i in range(n):
                    step_list += [[t, o, i + 1, -1, -1]]

        return step_list
    

    def update_mission_steps(self, step_list, nav_risk_mtx, picking_prob_mtx, throwing_prob_mtx):

        for step in step_list:
            time, risk = self.estimate_step_time_and_risk(step[1], step[0], nav_risk_mtx, picking_prob_mtx, throwing_prob_mtx)
            step[3] = time
            step[4] = risk

        return sorted(step_list, key=lambda x: (x[3], x[4]))


    def compute_qfa(self, step_list):

        qfa = np.zeros(self.scheduler_module.n_objects*(self.scheduler_module.n_trays+1), dtype=np.int64)

        for step in step_list:

            t = step[0]
            o = step[1]

            qfa[o * (self.scheduler_module.n_trays + 1)] += 1
            qfa[o * (self.scheduler_module.n_trays + 1) + t + 1] += 1

        return qfa

    def compute_mission_from_steps(self, steps):

        splitted_mission = {}
        for t in range(self.scheduler_module.n_trays):
            splitted_mission[self.trays[t]] = {}
            for o in range(self.scheduler_module.n_objects):
                splitted_mission[self.trays[t]][self.objects[o]] = 0

        for step in steps:
            splitted_mission[self.trays[step[0]]][self.objects[step[1]]] += 1

        return splitted_mission
    
    def update_final_state_from_mission_state(self, final_state, mission_state, t):

        for i in range(self.scheduler_module.n_objects):

            pick_mission_state = mission_state[2 + i * (self.scheduler_module.n_trays + 1)]

            place_mission_state = 0
            place_final_state = 0
            for j in range(self.scheduler_module.n_trays):

                index = 3 + i * (self.scheduler_module.n_trays + 1) + j
                final_state[index] += mission_state[index]

                place_mission_state += mission_state[index]
                place_final_state += final_state[index]

            final_state[2 + i * (self.scheduler_module.n_trays + 1)] = place_final_state + pick_mission_state - place_mission_state
        final_state[0] = t
        final_state[1] = mission_state[1]
        return final_state
    
        

    def solve_mission(self, missione):

        step_list = self.generate_mission_steps(missione)
        total_qfa = self.compute_qfa(step_list)

        print(f"total qfa {total_qfa}")
        self.ui_qfa_pub._publish_msg(total_qfa)
        
        t=0
        final_state = np.zeros(total_qfa.size + 2, dtype=np.int64)
        nav_risk_mtx,picking_prob_mtx,throwing_prob_mtx = self.retrieve_risk_mtx()
        step_list = self.update_mission_steps(step_list, nav_risk_mtx,picking_prob_mtx,throwing_prob_mtx)
        partial_mission = self.compute_mission_from_steps(step_list[:self.max_mission_steps])
        print("before solve mission internal")
        self.publish_active_mission(True)
        mission_state= self.solve_mission_internal(partial_mission, nav_risk_mtx, picking_prob_mtx, throwing_prob_mtx, final_state.copy())

        print(f"mission state {mission_state}")

        t += mission_state[0]
        print(f"final state before update {final_state}")
        final_state = self.update_final_state_from_mission_state(final_state,mission_state, t)
        
        print(f"final state after update {final_state}")
        
        while not np.array_equal(final_state[2:], total_qfa):
            
            residuo = total_qfa - final_state[2:]
            # residuo = np.maximum(residuo, np.zeros(residuo.shape))
            new_mission, new_state = self.from_array_to_dict_mission(residuo)

            step_list = self.generate_mission_steps(new_mission)
            nav_risk_mtx,picking_prob_mtx,throwing_prob_mtx = self.retrieve_risk_mtx()
            step_list = self.update_mission_steps(step_list, nav_risk_mtx,picking_prob_mtx,throwing_prob_mtx)

            ## TODO DA TESTARE
            ## capire come mantenere lo step dell'oggetto a bordo e aggiungerne solo altri 4
            residual_step = None
            for o in range(self.scheduler_module.n_objects):
                if new_state[self.objects[o]] > 0:
                    print(f"ho a bordo un oggetto di tipo {self.objects[o]}")
                    residual_step = list(filter(lambda x: x[1] == o, step_list))[0]
                    step_list.remove(residual_step)
                    print(f"ho estratto lo step {residual_step}")

            if residual_step:
                partial_step_list = step_list[:self.max_mission_steps - 1] + [residual_step]
            else:
                partial_step_list = step_list[:self.max_mission_steps]

            print(f"la mia partial list è {partial_step_list}")

            partial_mission = self.compute_mission_from_steps(partial_step_list)
            mission_state = self.solve_mission_internal(partial_mission, nav_risk_mtx, picking_prob_mtx, throwing_prob_mtx, final_state.copy(), new_state)

            print(f"mission state {mission_state}")
            
            t += mission_state[0]
            print(f"final state before update {final_state}")
            final_state = self.update_final_state_from_mission_state(final_state, mission_state, t)
            print(f"final state after update {final_state}")
            
        self.publish_active_mission(False)

        # da valutare poi se ha senso questo return
        return t,final_state


    def from_array_to_dict_mission(self,mission_array):
        new_mission = {}
        new_state   = {}
        
        for o in range(self.scheduler_module.n_objects):
            new_state[self.objects[o]] = sum([mission_array[(self.scheduler_module.n_trays + 1) * o + t + 1] for t in range(self.scheduler_module.n_trays)]) - mission_array[(self.scheduler_module.n_trays + 1) * o]
        
        for t in range(self.scheduler_module.n_trays):
            new_mission[self.trays[t]] = {}
            for o in range(self.scheduler_module.n_objects):
                # new_mission[self.trays[t]][self.objects[o]] = mission_array[self.scheduler_module.n_trays*o+t+1]
                new_mission[self.trays[t]][self.objects[o]] = mission_array[(self.scheduler_module.n_trays + 1) * o + t + 1]
        return new_mission, new_state

    def retrieve_risk_mtx(self):
        # richiesta calcolo rischi
        self.risk_estimation_request_pub._publish_msg(True)
        done = False
        while not done:
            if not self.risk_estimation_done_sub._check_empty_data():
                rospy.sleep(0.1)
                nav_risk_mtx   = self.navigation_risk_sub._risk_data
                pick_risk_mtx  = self.picking_risk_sub._risk_data
                throw_risk_mtx = self.throwing_risk_sub._risk_data
                self.risk_estimation_done_sub._reset_data()
                done=True
        return nav_risk_mtx,pick_risk_mtx,throw_risk_mtx
    
    def publish_scenarios(self, scenarios_lst, prob_lst):

        scenario_list = []

        for scenario_data in scenarios_lst:
            states = []

            for state_data in scenario_data:
                # Usa il nome del campo corretto, che è 'state' in base alla definizione di State
                state_msg = State(state=state_data)
                states.append(state_msg)

            # Usa il nome del campo corretto, che è 'state_list' in base alla definizione di Scenario
            scenario_msg = Scenario(state_list=states)
            scenario_list.append(scenario_msg)
        scenario_list = ScenarioList(scenario_list=scenario_list, probabilities=prob_lst)

        self.scenarios_pub._publish_msg(scenario_list)

        # Verifica se scenarios_lst è un'istanza di ScenarioList
        # scenario = Scenario(scenarios_lst)
        # serializzare gli oggetti in input in termini di ScenariosList e Float64MultiArray
        # andare a vedere come si fa a pubblicare con ros
        # self.scenarios_pub._publish_msg(...)
        # self.probabilities_pub._publish_msg(...)

    def publish_current_action(self, next_task):
        
        first_task = next_task['first_task']
        second_task = next_task['second_task']

        # Riempiamo il current task
        first_task_str = [str(first_task.get(key, '')) for key in first_task.keys()]
        # Assicurati che la lunghezza sia sempre 3
        first_task_str.extend([''] * (3 - len(first_task_str)))

        # Riempiamo il next task
        second_task_str = [str(second_task.get(key, '')) for key in second_task.keys()]
        # Assicurati che la lunghezza sia sempre 3
        second_task_str.extend([''] * (3 - len(second_task_str)))

        first_task_msg = Action(action=first_task_str)
        second_task_msg = Action(action=second_task_str)
        current_action = CurrentAction(first_action=first_task_msg, second_action=second_task_msg)

        self.ui_current_action_pub._publish_msg(current_action)
        
    def publish_current_state(self, final_state, mission_state):

        t = mission_state[0] + final_state[0]

        msg = self.update_final_state_from_mission_state(final_state.copy(), mission_state,t)

        self.ui_current_state_pub._publish_msg(msg)
        
        
        
        

    def publish_active_mission(self, value):
        self.mission_active_pub._publish_msg(value)
        return

    def publish_scenario_computation_done(self, value):
        self.scenario_computation_done_pub._publish_msg(value)
        return

    def solve_mission_internal(self, missione_internal, nav_risk_mtx,picking_prob_mtx,throwing_prob_mtx, final_state, new_state=None):
        mission_state= np.zeros(2+self.scheduler_module.n_objects*(self.scheduler_module.n_trays+1),dtype=np.int64)
        if new_state:
            for o in range(self.scheduler_module.n_objects):
                mission_state[(self.scheduler_module.n_trays + 1) * o + 2] = int(new_state[self.objects[o]])
    
        time_start_mission = time.perf_counter()
        print("before scheduling")
        self.scheduler_module.solve_mission(missione_internal,nav_risk_mtx,picking_prob_mtx,throwing_prob_mtx)
        print("print after scheduling")
        #Define state s0
        t = time.perf_counter() - time_start_mission
        # TODO go to safest closest node?
        # p,_ = self.get_closest_node(self.position_subscriber.get_position_xy(), with_safety=True)
        p = self.go_to_closest_node(with_safety=True)
        

        mission_state[0] = t
        mission_state[1] = p

        next_task, scenarios_lst, prob_lst = self.scheduler_module.next_task(mission_state)
        self.publish_current_action(next_task)
        rospy.loginfo(next_task)
        self.publish_scenarios(scenarios_lst, prob_lst)
        self.publish_scenario_computation_done(True)

        while not next_task['first_task']['action'] == "completed":

            if (next_task['first_task']['action'] == 'moving'):
                reschedule = self.perform_moving_task(next_task)
                
                if reschedule:
                    return mission_state ### TODO invertito
                mission_state, report = self.update_mission_state(mission_state,time_start_mission,True,next_task)
                self.publish_current_state(final_state, mission_state)
                self.send_report(report)
            else:
                success, reschedule = self.perform_manipulation_wait_drop_task(next_task)
                if reschedule:
                    return mission_state ### TODO invertito
                mission_state, report = self.update_mission_state(mission_state,time_start_mission,success,next_task)
                self.publish_current_state(final_state, mission_state)
                self.send_report(report)

            next_task, scenarios_lst, prob_lst = self.scheduler_module.next_task(mission_state)
            self.publish_current_action(next_task)
            rospy.loginfo(next_task)
            self.publish_scenarios(scenarios_lst, prob_lst)
            self.publish_scenario_computation_done(True)

        return mission_state
    
    def send_report(self, report):

        msg = Float64MultiArray()
        if report["action"] == "moving":

            msg.data = [
                report["starting_node"],
                report["ending_node"],
                report["delta_t"]
            ]
            self.send_navigation_report_pub._publish_msg(msg)

        elif report["action"] == "picking":

            msg.data = [
                report["starting_node_x"],
                report["starting_node_y"],
                report["box_x"],
                report["box_y"],
                float(report["success"])
            ]
            self.send_picking_report_pub._publish_msg(msg)
        
        elif report["action"] == "placing":

            msg.data = [
                report["starting_node_x"],
                report["starting_node_y"],
                report["tray_x"],
                report["tray_y"],
                float(report["success"])
            ]
            self.send_placing_report_pub._publish_msg(msg)

            
    def submit_goal(self,xp,yp,xo,yo,zo,wo):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = xp
        goal.target_pose.pose.position.y = yp
        goal.target_pose.pose.position.z = 0

        goal.target_pose.pose.orientation.x = xo
        goal.target_pose.pose.orientation.y = yo
        goal.target_pose.pose.orientation.z = zo
        goal.target_pose.pose.orientation.w = wo
        
        self.client.send_goal(goal)

        while not self.client.wait_for_result(timeout=rospy.Duration(0.01)):
            reschedule = self.check_for_reschedule()
            if reschedule:
                return True
            
        return False

        # TODO send navigation report (missing starting node)
    
    def send_moving_goal(self,xp,yp,xt,yt):

        angle = self.get_alpha_orientation(xp,yp,xt,yt)
        xo,yo,zo,wo = tf.transformations.quaternion_from_euler(0,0,angle)
        return self.submit_goal(xp,yp,xo,yo,zo,wo)

    def get_alpha_orientation(self,x_robot,y_robot,x_target,y_target):
        epsilon = 10**-3
        delta_x = x_target - x_robot
        delta_y = y_target - y_robot
        if (delta_x == 0):
            delta_x += epsilon
        return math.atan2(delta_y,delta_x)

    def perform_moving_task(self,next_task):

        target_node = next_task['first_task']['position']
        target_node_pos = self.action_graph_nodes[self.action_nodes[target_node]]

        if (next_task['second_task']['action'] == 'picking'):
            object_to_pick_int =  next_task['second_task']['object']
            object_to_pick = self.objects[object_to_pick_int]
            target_box = self.objects_box[object_to_pick]
            target_box_pos = self.location_coordinates[target_box]
            return self.send_moving_goal(target_node_pos['x'],target_node_pos['y'],target_box_pos['x'],target_box_pos['y'])

        if (next_task['second_task']['action'] == 'placing'):
            target_tray_int =  next_task['second_task']['tray']
            target_tray     = self.trays[target_tray_int]
            target_tray_pos = self.location_coordinates[target_tray]
            return self.send_moving_goal(target_node_pos['x'],target_node_pos['y'],target_tray_pos['x'],target_tray_pos['y'])
        
        return self.send_moving_goal(target_node_pos['x'],target_node_pos['y'],target_node_pos['x'],target_node_pos['y'])

    def perform_manipulation_wait_drop_task(self,next_task):

        self.manipulation_done_sub._reset_data()
        self.manipulation_success_sub._reset_data()

        if next_task['first_task']['action'] == 'picking':

            self.manipulation_action_type_pub._publish_msg("Pick")
            self.target_object_pub._publish_msg(self.objects[next_task['first_task']['object']])

            while self.manipulation_done_sub._check_empty_data():
                if self.check_for_reschedule():
                    return False, True
                rospy.sleep(0.1)

        elif next_task['first_task']['action'] == 'placing':

            self.manipulation_action_type_pub._publish_msg("Throw")
            self.target_object_pub._publish_msg(self.objects[next_task['first_task']['object']])
            self.target_tray_pub._publish_msg(self.trays[next_task['first_task']['tray']])

            while self.manipulation_done_sub._check_empty_data():
                if self.check_for_reschedule():
                    return False, True
                rospy.sleep(0.1)
                    
        elif next_task['first_task']['action'] == 'dropping':
            self.manipulation_action_type_pub._publish_msg("Drop")
            self.target_object_pub._publish_msg(self.objects[next_task['first_task']['object']])

            while self.manipulation_done_sub._check_empty_data():
                if self.check_for_reschedule():
                    return False, True
                rospy.sleep(0.1)
        else:   #waiting
            rospy.sleep(self.params["wait_time"])
        return self.manipulation_success_sub._data, False

    def update_mission_state(self,mission_state,time_start_mission,success,next_task):

        action_type = next_task["first_task"]["action"]

        t = time.perf_counter() - time_start_mission
        p,_ = self.get_closest_node(self.position_subscriber.get_position_xy())

        mission_state[0] = t
        mission_state[1] = p

        report = None
        if action_type == "moving":
            report = {
                "action": action_type,
                "starting_node": mission_state[1],
                "ending_node": p,
                "delta_t": t - mission_state[0]
            }
        elif action_type == "picking":
            obj_idx = next_task["first_task"]["object"]
            report = {
                "action": action_type,
                "starting_node_x": self.action_graph_nodes_int[mission_state[1]]["x"],
                "starting_node_y": self.action_graph_nodes_int[mission_state[1]]["y"],
                "box_x": self.location_coordinates[self.objects_box[self.objects[obj_idx]]]["x"],
                "box_y": self.location_coordinates[self.objects_box[self.objects[obj_idx]]]["y"],
                "success": success
            }
            if success:
                mission_state[2 + (1 + self.scheduler_module.n_trays) * obj_idx] += 1
        elif action_type == "placing":
            obj_idx = next_task["first_task"]["object"]
            tray_idx = next_task["first_task"]["tray"]
            report = {
                "action": action_type,
                "starting_node_x": self.action_graph_nodes_int[mission_state[1]]["x"],
                "starting_node_y": self.action_graph_nodes_int[mission_state[1]]["y"],
                "tray_x": self.location_coordinates[self.trays[tray_idx]]["x"],
                "tray_y": self.location_coordinates[self.trays[tray_idx]]["y"],
                "success": success
            }
            if success:
                mission_state[2 + (1 + self.scheduler_module.n_trays) * obj_idx + tray_idx + 1] += 1
            else:
                mission_state[2 + (1 + self.scheduler_module.n_trays) * obj_idx] -= 1
        elif action_type == "dropping":
            obj_idx = next_task["first_task"]["object"]
            report = {
                "action": action_type
            }
            if success:
                mission_state[2 + (1 + self.scheduler_module.n_trays) * obj_idx] -= 1
        elif action_type == "waiting":
            report = {
                "action": action_type
            }

        return mission_state, report