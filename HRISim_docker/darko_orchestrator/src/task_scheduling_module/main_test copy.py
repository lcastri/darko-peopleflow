import itertools
import json
import numba as nb
import numpy as np

class Test:
    def __init__(self):
        # load static data
        path_to_static_data = "../static_data"

        static_data_names = ["location_coordinates",
                                "action_graph_nodes",
                                "action_graph_conversion_dict",
                                "objects_box",
                                "rewards",
                                "action_times",
                                "items", "parameters"]

        static_data = {}

        for static_data_name in static_data_names:
            with open(path_to_static_data+"/"+static_data_name+".json", "r") as json_file:
                static_data[static_data_name] = json.load(json_file)

        self.action_nodes            = list(static_data["action_graph_nodes"].keys())
        self.n_action_nodes          = len(self.action_nodes)
        self.trays                   = static_data['items']['trays']
        self.objects                 = static_data['items']['objects']
        self.rewards                 = static_data['rewards']
        self.action_times            = static_data['action_times']
        self.action_graph_nodes      = static_data['action_graph_nodes'] 
        self.objects_box             = static_data['objects_box']
        self.location_coordinates    = static_data['location_coordinates']
        self.action_graph_nodes_int  = {static_data['action_graph_conversion_dict'][n]:static_data['action_graph_nodes'][n] for n in self.action_nodes }
        self.action_nodes_int        = list(self.action_graph_nodes_int.keys())
        self.parameters              = static_data["parameters"]
        self.t_horizon=120
        self.o_max=1
        self.n_trays=len(self.trays)
        self.n_objects=len(self.objects)
        self.quantities = np.zeros((self.n_objects,self.n_trays),dtype=np.int64)
        self.quantities[0,0] = 1
        self.params = self.parameters
        self.navigation_risk_mtx = np.ones((len(self.action_nodes),len(self.action_nodes),2),dtype=np.float64)
        self.picking_prob_mtx    = np.ones((len(self.action_nodes),len(self.objects)),dtype=np.int64)
        self.throwing_prob_mtx   = np.ones((len(self.action_nodes),len(self.trays)),dtype=np.int64)
        self.initialize_V()
        self.states = self.get_states()
        self.picking_reward = self.rewards['pick']
        self.placing_reward = self.rewards['place']
        self.picking_time   = self.action_times['pick']
        self.throwing_time  = self.action_times['throw']
        self.alpha_risk=0.01

        # _backward_pass(self.V,3,self.states,self.coeff,self.qft,self.quantities,2,2,self.picking_reward,self.placing_reward,self.picking_time,self.throwing_time,
        #                 self.navigation_risk_mtx,self.picking_prob_mtx,self.throwing_prob_mtx,self.alpha_risk,self.t_horizon,self.o_max)
        
        
        _backward_pass(self.V,len(self.action_nodes),self.states,self.coeff,self.qft,self.quantities,len(self.objects),len(self.trays),self.picking_reward,self.placing_reward,self.picking_time,self.throwing_time,
                        self.navigation_risk_mtx,self.picking_prob_mtx,self.throwing_prob_mtx,self.alpha_risk,self.t_horizon,self.o_max)
        print('ho finito')
                            
    def initialize_V(self):
        T = self.t_horizon
        P = len(self.action_nodes) + 1
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

@nb.njit
def _backward_pass(V,n_action_nodes,states,coeff,qft,quantities,n_objects,n_trays,picking_reward,placing_reward,picking_time,throwing_time,
                navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk,t_horizon,o_max):
    for t in range(t_horizon -1, -1, -1):
        for p in range(n_action_nodes):
            for q in states:
                s0 = (t,) + (p,) + tuple(q)
                if q == qft:
                    V[s0] = _final_state_value(s0, quantities, t_horizon, n_objects, n_trays)
                else:
                    #aggiungere azioni attesa e perdita oggetto
                    p1,r1s,s1s,r1f,s1f = _available_actions(s0,t_horizon,o_max,n_action_nodes,n_objects,n_trays,
                                                            picking_reward,placing_reward,picking_time,throwing_time,
                                                            quantities,navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk)
                    if (len(p1) == 0):
                        V[s0] = _final_state_value(s0, quantities, t_horizon, n_objects, n_trays)
                    else:
                        V[s0] = _get_max_value(V,coeff,p1,r1s,s1s,r1f,s1f)                        
                            
@nb.njit
def _available_actions(state,t_horizon,max_objects,n_action_nodes,n_objects,n_trays,picking_reward,placing_reward,picking_time,throwing_time,quantities,navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk):

    max_moving_actions   = n_action_nodes-1
    max_picking_actions  = n_objects
    max_throwing_actions = n_trays*n_objects
    max_actions          = max_moving_actions + max_picking_actions + max_throwing_actions

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
            delta_time,risk = navigation_risk_mtx[n0,n,1], navigation_risk_mtx[n0,n,3]
            cond_1 = (state[0] + delta_time) < t_horiz # controllo di essere dentro l'orizzonte temporale alla fine dell'azione di moving
            if cond_1:
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
            cond_1 = (state[0] + picking_time) < t_horiz # controllo di essere dentro l'orizzonte temporale alla fine dell'azione di picking
            cond_2 = state[(n_trays+1)*o+2]<max_picking[o] # controllo di non aver già preso tutti gli oggetti di quel tipo
            cond_3 = picking_prob_mtx[n0,o] > 0 # controllo che la probabilità di fare picking sia non nulla
            if cond_1 & cond_2 & cond_3:
                p1_ay[ns] = picking_prob_mtx[n0,o]
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
                cond_1 = (state[0] + throwing_time) < t_horiz # controllo di essere dentro l'orizzonte temporale alla fine dell'azione di throwing
                cond_2 = state[(n_trays+1) * o + 2] > (state[(n_trays+1) * o + 3] + state[(n_trays+1) * o + 4]) # controllo di avere almeno un oggetto di tipo o sul vassio
                cond_3 = state[(n_trays+1) * o + 3 + t] < quantities[o,t] # controllo di dover ancora posizionare oggetti di tipo o nel vassoio t
                cond_4 = throwing_prob_mtx[n0,t] > 0 # controllo che la probabilità di lanciare nel vassio t sia non nulla
                if (cond_1 & cond_2 & cond_3 & cond_4):
                    p1_ay[ns] = throwing_prob_mtx[n0,t]
                    # success state
                    r1_A_ay[ns] =  placing_reward  
                    s1_A_mtx[ns, :] = state[:]
                    s1_A_mtx[ns, 0] = state[0] + throwing_time
                    s1_A_mtx[ns, (n_trays+1)*o+3+t] += 1
                    # fail state
                    r1_B_ay[ns] = -picking_reward  
                    s1_B_mtx[ns, :] = state[:]
                    s1_B_mtx[ns, 0] = state[0] + throwing_time
                    s1_B_mtx[ns, (n_trays+1)*o+2] -= 1
                    ns += 1  
    return p1_ay[:ns],r1_A_ay[:ns],s1_A_mtx[:ns,:],r1_B_ay[:ns],s1_B_mtx[:ns,:]
                    

            
@nb.njit
def _final_state_value(state, quantities, t_horizon, n_objects, n_trays): #returns an arbitrary score for a final state (no actions available)
    v = 0.0
    for o in range(n_objects):
        for t in range(n_trays):
            v = v - 15 * (quantities[o, t] - state[(n_trays+1) * o + 3 + t])
    v = v + 0.5 * (t_horizon - state[0])
    return v

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
def _return_index(shape, indexes, coeff): #utility function used to compute the index to access the "unrolled" V matrix by a single index idx (workaround to let numba work)
    idx = 0
    for s in range(len(shape)):
        idx += indexes[s]*coeff[s]
    return int(np.round(idx))

def _compute_coefficients(shape): #utility function used to compute the coefficients needed in "_return_index"
    coeff = np.zeros(len(shape))
    for s in range(len(shape)):
        coeff[s] = np.prod(shape[s+1:])
    return coeff

@nb.njit
def _check(state, n_objects, n_trays, o_max): #function used in _get_states to filter unnecessary states (with n_obj > o_max)
    n_obj = 0
    for o in range(n_objects):
        index = o*(n_trays+1)
        n_obj+= state[index]
        for tray in range(n_trays):
            n_obj-= state[index+tray+1]
    return n_obj <= o_max

if __name__ == '__main__':
    Test()
