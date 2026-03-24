# %%
import numpy  as np
import json


class RiskMonitoring:
    def __init__(self, static_data_path="../static_data"):

        self.scenario_lst = None
        self.probability_lst = None
        self.updated_nav_risk_mtx   = None
        self.updated_pick_risk_mtx  = None
        self.updated_throw_risk_mtx = None
        self.nav_risk_mtx   = None
        self.pick_risk_mtx  = None
        self.throw_risk_mtx = None
        self.t_weight = 0.2
        self.r_weight = 0.8
        self.max_deviation = 0.20 # percentage variation
    
        with open(static_data_path + "/items.json", "r") as json_file:
            self.items = json.load(json_file)

        self.n_objects = len(self.items["objects"])
        self.n_trays = len(self.items["trays"])

    # order scenarios by probability
    def order_scenarios(self):
        scenarios_with_prob = list(zip(self.probability_lst, self.scenario_lst))
        ordered_scenarios_with_prob = sorted(scenarios_with_prob, key=lambda x: x[0], reverse=True)
        prob, scenarios = zip(*ordered_scenarios_with_prob) #unpacking
        return prob, scenarios

    # For each scenario, three lists are created to keep track of navigation(+ wait and drop) and picking and throwing actions respectively. 
    # For each action, the couple (node_from, node_to) is saved.
    # Finally, for each type of action, all lists of scenarios are collected in a single list.
    def filter_actions(self, ordered_scenarios):
        scenarios_nav_actions_lst = []
        scenarios_throw_actions_lst = []
        scenarios_pick_actions_lst = []
        for scenario in ordered_scenarios:
            navigation = []
            throwing = []
            picking = []
            for i in range(0, len(scenario)-1): # la lenght scenario è stabilita dal parametro "scenario lenght" nel file parameters.json letto dallo scheduler.py al momento della generazione degli scenari
                s0 = np.array(scenario[i])
                s1 = np.array(scenario[i+1])
                if s1[1] != s0[1]: # navigation
                    navigation.append((s0[1],s1[1]))
                else: # può essere wait/drop/place/pick
                    if np.array_equal(s1[2:],s0[2:]): # azione di wait, la tratto come un'azione di navigation (ma il nodo non cambia)
                        navigation.append((s0[1],s1[1]))
                        continue
                    diff = s1[2:]-s0[2:]
                    if np.min(diff) <0: # drop action, la tratto come un'azione di navigation (ma il nodo non cambia)
                        idx = np.argmin(diff)
                        obj = int(np.round(idx/(self.n_trays+1)))
                        navigation.append((s0[1],s1[1]))
                        continue
                    idx = np.argmax(diff) 
                    tray = idx%(self.n_trays+1)-1 # questo mi restituisce l'indice del vassoio relativo al pick/place
                    if tray == -1: # picking
                        obj=int(np.round(idx/(self.n_trays+1)))   
                        picking.append((s0[1], obj)) 
                        continue
                    throwing.append((s0[1],tray)) # placing
            scenarios_nav_actions_lst.append(navigation) 
            scenarios_throw_actions_lst.append(throwing) 
            scenarios_pick_actions_lst.append(picking) 
        return scenarios_nav_actions_lst, scenarios_throw_actions_lst, scenarios_pick_actions_lst

    # get the list of scenarios (where a scenario is a list of actions to do) 
    # and the list of corresponding probabilities
    def get_scenarios_and_probabilities(self):
        ordered_prob_lst, ordered_scenarios_lst = self.order_scenarios()
        scenarios_nav_actions_lst, scenarios_throw_actions_lst, scenarios_pick_actions_lst = self.filter_actions(ordered_scenarios_lst)
        return scenarios_nav_actions_lst, scenarios_throw_actions_lst, scenarios_pick_actions_lst, ordered_prob_lst

    # adjust the risk values for the ui
    def risk_to_ui(self, valore):
        if valore <= 15:
            return max(valore-5,0)
        elif valore <= 35:
            return valore - 10
        else:
            return min(valore + 15, 100)
        
    # evaluate the risk deviation comparing the previous navigation risk map with the new one
    def get_risk_deviation(self):
        scen_nav_act_lst, scen_throw_act_lst, scen_pick_act_lst,  ordered_prob_lst = self.get_scenarios_and_probabilities()
        # create new arrays
        nav_deltas = np.zeros(len(scen_nav_act_lst))
        throw_deltas = np.zeros(len(scen_throw_act_lst))
        pick_deltas = np.zeros(len(scen_pick_act_lst))
        nav_risk_tot_new = np.zeros(len(scen_nav_act_lst))
        nav_risk_tot_old = np.zeros(len(scen_nav_act_lst))
        throw_risk_tot_new = np.zeros(len(scen_throw_act_lst))
        throw_risk_tot_old = np.zeros(len(scen_throw_act_lst))
        pick_risk_tot_new = np.zeros(len(scen_throw_act_lst))
        pick_risk_tot_old = np.zeros(len(scen_throw_act_lst))
        num_scenarios = len(scen_nav_act_lst)
        # for each scenario, compute the risk deviation
        for i in range(0, num_scenarios): # itero per ogni scenario
            nav_delta_t, nav_delta_r = 0, 0
            nav_old_t, nav_old_r, nav_new_t, nav_new_r  = 0, 0, 0, 0
            throw_delta_r, pick_delta_r = 0, 0
            throw_old_r, throw_new_r  = 0, 0
            pick_old_r, pick_new_r  = 0, 0
            for action in scen_nav_act_lst[i]:
                # sum the risk and time values for all the navigation actions of the scenario (nav, wait and drop actions)
                nav_old_t += self.nav_risk_mtx[action][1] # add old t
                nav_old_r += self.nav_risk_mtx[action][3] # add old r
                nav_new_t += self.updated_nav_risk_mtx[action][1] # add new t
                nav_new_r += self.updated_nav_risk_mtx[action][3] # add new r
            for action in scen_throw_act_lst[i]:
                # sum the risk values (success probabilities) for all the throwing actions of the scenario
                throw_old_r += 100 - self.throw_risk_mtx[action] # add old r
                throw_new_r += 100 - self.updated_throw_risk_mtx[action] # add new r
            for action in scen_pick_act_lst[i]:
                # sum the risk values (success probabilities) for all the picking actions of the scenario
                pick_old_r += 100 - self.pick_risk_mtx[action] # add old r
                pick_new_r += 100 - self.updated_pick_risk_mtx[action] # add new r 

            # total time/risk variation for the scenario
            # navigation
            nav_delta_t = ((nav_new_t - nav_old_t)/(nav_old_t + 0.01))*((nav_new_t-nav_old_t)/5)*100
            nav_delta_r = ((nav_new_r - nav_old_r)/(nav_old_r + 0.01))*((nav_new_r-nav_old_r)/5)*100
            nav_deltas[i] = self.t_weight*(nav_delta_t) + self.r_weight*(nav_delta_r)

            nav_risk_tot_new[i] = self.t_weight*(nav_new_t) + self.r_weight*(nav_new_r)
            nav_risk_tot_old[i] = self.t_weight*(nav_old_t) + self.r_weight*(nav_old_r)

            # throwing
            throw_delta_r = abs(throw_new_r - throw_old_r)/(throw_old_r + 0.01)*100
            throw_deltas[i] = throw_delta_r

            throw_risk_tot_new[i] = throw_new_r
            throw_risk_tot_old[i] = throw_old_r
            
            # picking
            pick_delta_r = abs(pick_new_r - pick_old_r)/(pick_old_r + 0.01)*100
            pick_deltas[i] = pick_delta_r
            
            pick_risk_tot_new[i] = pick_new_r
            pick_risk_tot_old[i] = pick_old_r

        prob_ay = np.array(ordered_prob_lst)

        # weighted sum of the risk deltas of each scenario for the scenarios' probabilities
        nav_delta_tot = np.dot(nav_deltas, prob_ay)/100 
        throw_delta_tot = np.dot(throw_deltas, prob_ay)/100
        pick_delta_tot = np.dot(pick_deltas, prob_ay)/100
        # weighted sum of the total risk values of each scenario for the scenarios' probabilities (adjusted for ui)
        nav_old_tot = self.risk_to_ui(np.dot(nav_risk_tot_old, prob_ay))
        nav_new_tot = self.risk_to_ui(np.dot(nav_risk_tot_new, prob_ay))
        throw_old_tot = self.risk_to_ui(np.dot(throw_risk_tot_old, prob_ay))
        throw_new_tot = self.risk_to_ui(np.dot(throw_risk_tot_new, prob_ay))
        pick_old_tot = self.risk_to_ui(np.dot(pick_risk_tot_old, prob_ay))
        pick_new_tot = self.risk_to_ui(np.dot(pick_risk_tot_new, prob_ay))

        return nav_delta_tot, throw_delta_tot, pick_delta_tot, nav_old_tot, nav_new_tot, throw_old_tot, throw_new_tot, pick_old_tot, pick_new_tot

    # trigger the re-schedule if the risk deviation is greater than a given threshold
    # return the boolean value to trigger the reschedule and the risk values to show in the ui
    def reschedule_evaluation(self):
        nav_delta_tot, throw_delta_tot, pick_delta_tot, nav_old_tot, nav_new_tot, throw_old_tot, throw_new_tot, pick_old_tot, pick_new_tot = self.get_risk_deviation() # percentage variation
        max_deviation = max(nav_delta_tot, throw_delta_tot, pick_delta_tot)
        manip_old_tot = max(throw_old_tot, pick_old_tot)
        manip_new_tot = max(throw_new_tot, pick_new_tot)
        if max_deviation > self.max_deviation:
            return True, nav_old_tot, nav_new_tot, manip_old_tot, manip_new_tot
        
        return False, nav_old_tot, nav_new_tot, manip_old_tot, manip_new_tot


