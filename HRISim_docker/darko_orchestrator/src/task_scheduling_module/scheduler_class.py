import numpy as np
import numba as nb
import itertools

class Scheduler:
    def __init__(self, params,action_nodes,trays,objects,rewards,action_times,t_list):

        self.t_horizon=params["t_horizon"]
        self.alpha_risk = params["alpha_risk"]
        self.o_max=params["o_max"]
        self.min_prob=params["min_prob"]
        self.scenario_length=params["scenario_length"]
        
        self.action_nodes = action_nodes
        self.n_action_nodes = len(action_nodes)
        self.trays = trays
        self.objects = objects 
        self.n_objects = len(objects)
        self.n_trays   = len(trays)

        self.quantities = np.zeros((self.n_objects,self.n_trays),dtype=np.int64)
        self.quantities[0,0] = 1

        self.navigation_risk_mtx = np.ones((self.n_action_nodes,self.n_action_nodes,4),dtype=np.float64)
        self.picking_prob_mtx    = np.ones((self.n_action_nodes,self.n_objects),dtype=np.int64)
        self.throwing_prob_mtx   = np.ones((self.n_action_nodes,self.n_trays),dtype=np.int64)

        self.picking_reward = rewards['pick']
        self.placing_reward = rewards['place']
        self.wait_reward = rewards['wait']
        self.picking_time   = action_times['pick']
        self.throwing_time  = action_times['throw']
        self.wait_time = action_times["wait"]
        self.drop_time = action_times["drop"]


        self.t_list = t_list

        self.initialize_V()
        states = self.get_states()
        self.V = _backward_pass(self.V,self.n_action_nodes,states,self.coeff,self.qft,self.quantities,
                                self.n_objects,self.n_trays,self.picking_reward,self.placing_reward,self.wait_reward,self.picking_time,self.throwing_time,self.wait_time,self.drop_time,
                                self.navigation_risk_mtx,self.picking_prob_mtx,self.throwing_prob_mtx,self.alpha_risk,self.t_horizon,self.o_max, self.t_list)

    def solve_mission(self,missione,navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx):

        self.navigation_risk_mtx = navigation_risk_mtx
        self.picking_prob_mtx    = picking_prob_mtx
        self.throwing_prob_mtx   = throwing_prob_mtx

        for t in range(self.n_trays):
            for o in range(self.n_objects):
                self.quantities[o,t] = missione[self.trays[t]][self.objects[o]]
        self.initialize_V()
        states = self.get_states()
        self.V = _backward_pass(self.V,self.n_action_nodes,states,self.coeff,self.qft,self.quantities,
                                self.n_objects,self.n_trays,self.picking_reward,self.placing_reward,self.wait_reward,self.picking_time,self.throwing_time,self.wait_time,self.drop_time,
                                self.navigation_risk_mtx,self.picking_prob_mtx,self.throwing_prob_mtx,self.alpha_risk,self.t_horizon,self.o_max, self.t_list)

    def initialize_V(self):
        T = self.t_horizon
        P = self.n_action_nodes + 1
        q = self.quantities

        sizeV = [T, P] + list(itertools.chain.from_iterable(
            [1 + sum(q[o, t] for t in range(self.n_trays))] + [q[o, t] + 1 for t in range(self.n_trays)] for o in
            range(self.n_objects)))
        self.V = np.zeros(sizeV, dtype="i8")
        self.coeff = _compute_coefficients(self.V.shape)
        self.qfa = np.asarray(self.V.shape[2:],dtype="i8")-1
        self.qft = tuple(self.qfa)

    def get_states(self):
        lobjects = []
        for o in range(self.n_objects):
            iterable = [1+sum(self.quantities[o,t] for t in range(self.n_trays))]+[self.quantities[o,t]+1 for t in range(self.n_trays)]
            lobjects.append(list(filter(lambda x : x[0]>=sum(x[1:]),itertools.product(*[range(i) for i in iterable]))))
        return nb.typed.List(filter(lambda s: _check(s, self.n_objects, self.n_trays, self.o_max),
                                    [tuple(itertools.chain.from_iterable(t)) for t in itertools.product(*lobjects)]))
        
    def available_actions(self,state):

        p1,r1s,s1s,r1f,s1f = _available_actions(state,self.t_horizon,self.o_max,self.n_action_nodes,self.n_objects,self.n_trays,
                                                     self.picking_reward,self.placing_reward,self.wait_reward,self.picking_time,self.throwing_time,self.wait_time,self.drop_time,
                                                     self.quantities,self.navigation_risk_mtx,self.picking_prob_mtx,self.throwing_prob_mtx,
                                                     self.alpha_risk, self.t_list)
        return p1,r1s,s1s,r1f,s1f
    
    def compute_scenarios(self, state):
        scenarios_lst, open_scenarios_lst =  [], []
        prob_lst, open_prob_lst = [], []
        open_scenarios_lst.append([state])
        open_prob_lst.append(1)
        while len(open_scenarios_lst) > 0:
            current_scenario = open_scenarios_lst[0]
            current_scenario_prob = open_prob_lst[0]
            closed_scenario = False
            while not closed_scenario:
                s0 = current_scenario[-1]  # last state added to the scenario
                # print('sto chiamando da compute scenarios')
                p1_ay,r1_A_ay,s1_A_mtx,r1_B_ay,s1_B_mtx = self.available_actions(s0)
                if len(p1_ay)>0 and len(current_scenario)<self.scenario_length:
                    a_star = self.get_idx_max_value(1,p1_ay,r1_A_ay,s1_A_mtx,r1_B_ay,s1_B_mtx) # -> computes index of the best action
                    success_prob = p1_ay[a_star]/100
                    if success_prob < 1:  # new branch evaluation
                        # case 1: a_star fails
                        prob_new_branch = current_scenario_prob*(1-success_prob)
                        if prob_new_branch >= self.min_prob:
                            new_branch_scenario = current_scenario[:] # a new scenario is created
                            new_branch_scenario.append(np.asarray(s1_B_mtx[a_star,:],dtype=np.int64)) # add next state
                            open_scenarios_lst.append(new_branch_scenario) # save the new scenario
                            open_prob_lst.append(prob_new_branch) # save the probability
                        # case 1: a_star succeeds
                        current_scenario.append(np.asarray(s1_A_mtx[a_star,:],dtype=np.int64))  # add next state
                        updated_prob = current_scenario_prob*success_prob  # compute new scenario probability
                        if updated_prob < self.min_prob: # close scenario if prob is low
                            open_scenarios_lst.pop(0)
                            scenarios_lst.append(current_scenario)
                            open_prob_lst.pop(0)
                            prob_lst.append(round(updated_prob, 3))
                            closed_scenario = True
                        current_scenario_prob = updated_prob # update probability for current scenario
                    else: # only success case
                        current_scenario.append(np.asarray(s1_A_mtx[a_star,:],dtype=np.int64))
                        current_scenario_prob = current_scenario_prob*success_prob
                else: # close scenario
                    open_scenarios_lst.pop(0)
                    scenarios_lst.append(current_scenario)
                    open_prob_lst.pop(0)
                    prob_lst.append(round(current_scenario_prob, 3))
                    closed_scenario = True
        return scenarios_lst, prob_lst


    def next_task(self,state):
        scenarios_lst, prob_lst = self.compute_scenarios(state)

        typ,new_state,next_state = _forward_step(state,self.V,self.coeff,self.qfa,self.t_horizon,self.o_max,self.n_action_nodes,self.n_objects,self.n_trays,self.picking_reward,
                                                 self.placing_reward,self.wait_reward,self.picking_time,self.throwing_time,self.wait_time,self.drop_time,self.quantities,self.navigation_risk_mtx,
                                                 self.picking_prob_mtx,self.throwing_prob_mtx,self.alpha_risk, self.t_list)
        
        return self.generate_task(state,typ,new_state,next_state), scenarios_lst, prob_lst
    
    def generate_task(self,state,typ,new_state,next_state):
    #     # 0 -> end
    #     # 1 -> pick/place/wait/drop
    #     # 2 -> move
        task = {"first_task":{},
                "second_task":{}}
        if (typ == 1):
            act,obj,tray = _pick_place_wait_drop_action(state,new_state,self.n_trays)
            if (act == 'picking'):
                task['first_task']['action'] = act
                task['first_task']['object'] = obj
            elif act=='placing':
                task['first_task']['action'] = act
                task['first_task']['object'] = obj
                task['first_task']['tray']   = tray
            elif act=='waiting':
                task['first_task']['action'] = act
            elif act=='dropping':
                task['first_task']['action'] = act
                task['first_task']['object'] = obj
            else:
                task['first_task']['action'] = 'error'
        elif (typ == 2):
            task['first_task']['action'] = 'moving'
            task['first_task']['position'] = new_state[1]
            act,obj,tray = _pick_place_wait_drop_action(new_state,next_state,self.n_trays)
            if (act == 'picking'):
                task['second_task']['action'] = act
                task['second_task']['object'] = obj
            elif act=='placing':
                task['second_task']['action'] = act
                task['second_task']['object'] = obj
                task['second_task']['tray']   = tray
            elif act=='waiting':
                task['second_task']['action'] = act
            elif act=='dropping':
                task['second_task']['action'] = act
                task['second_task']['object'] = obj
            else:
                task['second_task']['action'] = 'error'
        else:
            task['first_task']['action'] = "completed"
        return task
    
    def get_idx_max_value(self,discount_factor,p1,r1s,s1s,r1f,s1f):
        return _get_idx_max_value(discount_factor,self.V,self.coeff,p1,r1s,s1s,r1f,s1f)
    

def _pick_place_wait_drop_action(s0,s1,n_trays):
    
    if np.array_equal(s1[2:],s0[2:]): #azione di wait
        return "waiting",-1,-1
    
    state_diff = s1[2:]-s0[2:]
    
    if np.min(state_diff) <0: #drop action
        idx = np.argmin(state_diff)
        obj = int(np.round(idx/(n_trays+1)))
        return "dropping", obj, -1
    
    idx = np.argmax(state_diff)
    tray = idx%(n_trays+1)-1 #questo mi restituisce l'indice del vassoio relativo al pick/place
    if tray == -1:
        obj=int(np.round(idx/(n_trays+1)))
        return "picking",obj,-1
    
    obj = int(np.round((idx-tray-1)/(n_trays+1)))
    return "placing",obj,tray

@nb.njit
def _check(state, n_objects, n_trays, o_max): #function used in _get_states to filter unnecessary states (with n_obj > o_max)
    n_obj = 0
    for o in range(n_objects):
        index = o*(n_trays+1)
        n_obj+= state[index]
        for tray in range(n_trays):
            n_obj-= state[index+tray+1]
    return n_obj <= o_max

@nb.njit
def _final_state_value(state, quantities, t_horizon, n_objects, n_trays): #returns an arbitrary score for a final state (no actions available)
    v = 0.0
    for o in range(n_objects):
        for t in range(n_trays):
            v = v - 15 * (quantities[o, t] - state[(n_trays+1) * o + 3 + t])
    v = v + 0.5 * (t_horizon - state[0])
    return v

def _compute_coefficients(shape): #utility function used to compute the coefficients needed in "_return_index"
    coeff = np.zeros(len(shape))
    for s in range(len(shape)):
        coeff[s] = np.prod(shape[s+1:])
    return coeff

@nb.njit
def _return_index(shape, indexes, coeff): #utility function used to compute the index to access the "unrolled" V matrix by a single index idx (workaround to let numba work)
    idx = 0
    for s in range(len(shape)):
        idx += indexes[s]*coeff[s]
    return int(np.round(idx))

@nb.njit
def _interp_nav_time_risk(nav_mtx, n1, n2, t, t_list):

    idx = 0
    while idx < len(t_list) and t_list[idx] < t:
        idx += 1

    idx = min(idx, len(t_list) - 1)

    return max(1, nav_mtx[n1, n2, 4 * idx + 1]), nav_mtx[n1, n2, 4 * idx + 3]

@nb.njit
def _interp_manip_risk(manip_mtx, n_elem, elem, n, t, t_list):

    idx = 0
    while idx < len(t_list) and t_list[idx] < t:
        idx += 1

    idx = min(idx, len(t_list) - 1)

    return manip_mtx[n, n_elem * idx + elem]


@nb.njit
def _available_actions(state,t_horizon,max_objects,n_action_nodes,n_objects,n_trays,picking_reward,placing_reward,wait_reward,picking_time,throwing_time,wait_time,drop_time,quantities,navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk,t_list):
    
    max_moving_actions   = n_action_nodes-1
    max_picking_actions  = n_objects
    max_throwing_actions = n_trays*n_objects
    max_waiting_actions  = 1
    max_dropping_actions = max_objects
    max_actions          = max_moving_actions + max_picking_actions + max_throwing_actions + max_waiting_actions + max_dropping_actions
    
    t_horiz,max_objs = t_horizon, max_objects
    max_picking = np.sum(quantities,axis=1)
    obj_on_tray = sum([state[(n_trays+1)*o+2]-(state[(n_trays+1)*o+3] + state[(n_trays+1)*o+4]) for o in range(n_objects)])  # DA MODIFICARE, generalizzando per numero vassoi

    p1_ay = np.zeros(max_actions, dtype='f8')  # create empty reward array
    r1_A_ay = np.zeros(max_actions, dtype='f8')  # create empty reward array
    r1_B_ay = np.zeros(max_actions, dtype='f8')  # create empty reward array
    s1_A_mtx = np.zeros((max_actions, len(state)), dtype='i8')  # create empty new_state mtx
    s1_B_mtx = np.zeros((max_actions, len(state)), dtype='i8')  # create empty new_state mtx

    ns = 0
    n0 = state[1]
    # Moving actions

    for n in range(n_action_nodes):

        if (n != n0):

            # rospy.sleep(20)

            # delta_time,risk = max(1,navigation_risk_mtx[n0,n,1]), navigation_risk_mtx[n0,n,3]
            delta_time, risk = _interp_nav_time_risk(navigation_risk_mtx, n0, n, state[0], t_list)

            cond_1 = (state[0] + delta_time) < t_horiz # controllo di essere dentro l'orizzonte temporale alla fine dell'azione di moving
            cond_2 = risk <= 10
            if cond_1 and cond_2:
                # print('n0 ', n0, ' n ', n, ' time ', delta_time, ' risk ', risk)
                p1_ay[ns] = 100
                # success state
                r1_A_ay[ns] = -alpha_risk*risk  
                s1_A_mtx[ns, :] = state[:]
                s1_A_mtx[ns, 0] = state[0] + delta_time
                s1_A_mtx[ns, 1] = n
                ns += 1  

    # Picking actions
    if (obj_on_tray<max_objs):

        for o in range(n_objects):

            picking_prob = _interp_manip_risk(picking_prob_mtx, n_objects, o, n0, state[0], t_list)

            cond_1 = (state[0] + picking_time) < t_horiz # controllo di essere dentro l'orizzonte temporale alla fine dell'azione di picking
            cond_2 = state[(n_trays+1)*o+2]<max_picking[o] # controllo di non aver già preso tutti gli oggetti di quel tipo
            cond_3 = picking_prob > 50 # controllo che la probabilità di fare picking sia non nulla

            if cond_1 and cond_2 and cond_3:

                p1_ay[ns] = picking_prob
                # success state
                r1_A_ay[ns] = picking_reward
                s1_A_mtx[ns, :] = state[:]  
                s1_A_mtx[ns, 0] = state[0] + picking_time
                s1_A_mtx[ns, (n_trays+1)*o+2] += 1

                # fail state
                r1_B_ay[ns] = 0  
                s1_B_mtx[ns, :] = state[:]
                s1_B_mtx[ns, 0] = state[0] + picking_time
                ns += 1  

    #  Throwing actions
    if (obj_on_tray > 0):

        for o in range(n_objects):

            for t in range(n_trays):

                throwing_prob = _interp_manip_risk(throwing_prob_mtx, n_trays, t, n0, state[0], t_list)

                cond_1 = (state[0] + throwing_time) < t_horiz # controllo di essere dentro l'orizzonte temporale alla fine dell'azione di throwing
                cond_2 = state[(n_trays+1) * o + 2] > (state[(n_trays+1) * o + 3] + state[(n_trays+1) * o + 4]) # controllo di avere almeno un oggetto di tipo o sul vassio
                cond_3 = state[(n_trays+1) * o + 3 + t] < quantities[o,t] # controllo di dover ancora posizionare oggetti di tipo o nel vassoio t
                cond_4 = throwing_prob > 50 # controllo che la probabilità di lanciare nel vassio t sia non nulla
                # print(f'Sto considerando il tray {t}')
                # print(f'Sto provando a throware dal nodo {state[1]} con una probabilità di {throwing_prob_mtx[n0,t]}')
                if (cond_1 and cond_2 and cond_3 and cond_4):
                    p1_ay[ns] = throwing_prob
                    # success state
                    r1_A_ay[ns] = placing_reward  
                    s1_A_mtx[ns, :] = state[:]
                    s1_A_mtx[ns, 0] = state[0] + throwing_time
                    s1_A_mtx[ns, (n_trays+1)*o+3+t] += 1
                    # fail state
                    r1_B_ay[ns] = -picking_reward  
                    s1_B_mtx[ns, :] = state[:]
                    s1_B_mtx[ns, 0] = state[0] + throwing_time
                    s1_B_mtx[ns, (n_trays+1)*o+2] -= 1
                    ns += 1  
    # Waiting actions
    _,risk = _interp_nav_time_risk(navigation_risk_mtx, n0, n0, state[0], t_list)
    cond= (state[0]+wait_time) < t_horiz
    if cond:
        p1_ay[ns]=100
        #success state
        r1_A_ay[ns] = wait_reward
        s1_A_mtx[ns, :] = state[:]
        s1_A_mtx[ns, 0] = state[0] + wait_time
        ns += 1 
    
    # Dropping actions
    _,risk = _interp_nav_time_risk(navigation_risk_mtx, n0, n0, state[0], t_list)
    if (obj_on_tray > 0):

        for o in range(n_objects):

            cond_1 = (state[0] + drop_time) < t_horiz # controllo di essere dentro l'orizzonte temporale alla fine dell'azione di dropping

            cond_2 = state[(n_trays+1) * o + 2] > sum([state[(n_trays+1) * o + 3 + t] for t in range(n_trays)]) # controllo di avere almeno un oggetto di tipo o sul vassoio

            if cond_1 and cond_2:

                p1_ay[ns] = 100
                # success state
                r1_A_ay[ns] = -picking_reward
                s1_A_mtx[ns, :] = state[:]
                s1_A_mtx[ns, 0] = state[0] + drop_time
                s1_A_mtx[ns, (n_trays+1)*o+2] -= 1
                # print(f"State: {state.tolist()}")
                # print(f"New state: {s1_A_mtx[ns].tolist()}")
                ns += 1
                
                
    return p1_ay[:ns],r1_A_ay[:ns],s1_A_mtx[:ns,:],r1_B_ay[:ns],s1_B_mtx[:ns,:]
                    

@nb.njit
def _get_max_value(V,coeff,p1,r1s,s1s,r1f,s1f):
    v_max_ay = np.zeros(len(p1))
    for a in range(len(p1)):
        prob_success        = p1[a]
        reward_success      = r1s[a]
        reward_fail         = r1f[a]
        value_state_success = np.take(V,_return_index(V.shape, np.asarray(s1s[a,:]),coeff))
        value_state_fail    = np.take(V,_return_index(V.shape, np.asarray(s1f[a,:]),coeff))
        v_max_ay[a] = prob_success/100*(reward_success+value_state_success) + (100-prob_success)/100*(reward_fail+value_state_fail)
    return np.max(v_max_ay)

@nb.njit
def _get_idx_max_value(discount_factor,V,coeff,p1,r1s,s1s,r1f,s1f):
    v_max_ay = np.zeros(len(p1))
    for a in range(len(p1)):
        prob_success        = p1[a]
        reward_success      = r1s[a]
        reward_fail         = r1f[a]
        value_state_success = discount_factor*np.take(V,_return_index(V.shape, np.asarray(s1s[a,:]),coeff))
        value_state_fail    = discount_factor*np.take(V,_return_index(V.shape, np.asarray(s1f[a,:]),coeff))
        v_max_ay[a] = prob_success/100*(reward_success+value_state_success) + (100-prob_success)/100*(reward_fail+value_state_fail)
    return np.argmax(v_max_ay)

@nb.njit
def _backward_pass(V,n_action_nodes,states,coeff,qft,quantities,n_objects,n_trays,picking_reward,placing_reward,wait_reward,picking_time,throwing_time,wait_time,drop_time,
                   navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk,t_horizon,o_max, t_list):
    for t in range(t_horizon -1, -1, -1):
        for p in range(n_action_nodes):
            for q in states:
                s0 = (t,) + (p,) + tuple(q)
                if q == qft:
                    V[s0] = _final_state_value(s0, quantities, t_horizon, n_objects, n_trays)
                else:
                    # print('sto chiamando da backward pass')
                    p1,r1s,s1s,r1f,s1f = _available_actions(s0,t_horizon,o_max,n_action_nodes,n_objects,n_trays,
                                                            picking_reward,placing_reward,wait_reward,picking_time,throwing_time,wait_time,drop_time,
                                                            quantities,navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk, t_list)
                    if (len(p1) == 0):
                        V[s0] = _final_state_value(s0, quantities, t_horizon, n_objects, n_trays)
                    else:
                        V[s0] = _get_max_value(V,coeff,p1,r1s,s1s,r1f,s1f)
    return V

@nb.njit
def _forward_step(state,V,coeff,qfa,t_horizon,o_max,n_action_nodes,n_objects,n_trays,picking_reward,placing_reward,wait_reward,picking_time,throwing_time,wait_time,drop_time,quantities,navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk, t_list):
    #     # 0 -> end
    #     # 1 -> pick/place/wait/drop
    #     # 2 -> move
    if np.array_equal(state[2:],qfa):
        return 0,state,state
    else:
        # print("inizio forward step")
        s_move = np.zeros(len(state),dtype=np.int64)
        s_move[:] = state[:]
    while True:
        # print("Sto chiamando dalla forward pass")
        p1,r1s,s1s,r1f,s1f = _available_actions(s_move,t_horizon,o_max,n_action_nodes,n_objects,n_trays,picking_reward,placing_reward,wait_reward,picking_time,throwing_time,wait_time,drop_time,quantities,navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk, t_list)
        # print(f"Tempo rimanente: {t_horizon-s_move[0]}")
        # print(f"Numero di azioni disponibili: {len(p1)}")
        if (len(p1)==0):
            # print("Ho zero azioni disponibili")
            if (state[0] > t_horizon*0.8):
                return 0, state, state
            return  1,state,state
        # print("nel mezzo inoltrato della forward step")
        # print(f"Stato corrente (s_move): {s_move.tolist()}")
        # print(f"Possibili stati di arrivo (s1s): {s1s.tolist()}")
        idx = _get_idx_max_value(0.95,V,coeff,p1,r1s,s1s,r1f,s1f)
        s_next = s1s[idx,:]
        if (s_next[1] != s_move[1]):
            # print("nel mezzo più inoltrato della forward step")
            # print("##############Tempo e rischio:")
            # print(f's_move[1]: {s_move[1]}')
            # print(f's_next[1]: {s_next[1]}')
            # print(navigation_risk_mtx[s_move[1]][s_next[1]])
            s_move[:] = s_next[:]
        else:
            # print("nella fine della forward step")
            # print(np.array_equal(state,s_move))
            if np.array_equal(state,s_move): #se non mi sono mai mosso
                # print("nell'ultimo if della forward step")
                return 1,s_next,state #azione successiva è manipolazione 
            else:
                # print("nell'else dell'ultimo if della forward step")
                return  2,s_move,s_next #mi sono mosso
