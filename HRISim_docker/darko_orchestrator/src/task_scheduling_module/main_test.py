#%%
import itertools
import json
import time
import numba as nb
import numpy as np
import pandas as pd

class Test:
    def __init__(self, missione):
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
        for t in range(self.n_trays):
            for o in range(self.n_objects):
                self.quantities[o,t] = missione[self.trays[t]][self.objects[o]]
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
    

    def measure_backward_pass_time(self, missione):
        start_time = time.time()
        _backward_pass(self.V, len(self.action_nodes), self.states, self.coeff, self.qft, self.quantities,
                            len(self.objects), len(self.trays), self.picking_reward, self.placing_reward,
                            self.picking_time, self.throwing_time, self.navigation_risk_mtx, self.picking_prob_mtx,
                            self.throwing_prob_mtx, self.alpha_risk, self.t_horizon, self.o_max)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for mission: {elapsed_time} seconds")
        return elapsed_time
                            
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
    # Waiting actions
    t_wait = 2
    cond= (state[0]+t_wait) < t_horiz
    if cond:
        p1_ay[ns]=100
        #success state
        r1_A_ay[ns] = 0
        s1_A_mtx[ns, :] = state[:]
        s1_A_mtx[ns, 0] = state[0] + t_wait
        ns += 1 
    
    # Dropping actions
    if (obj_on_tray>0):
        for o in range(n_objects):
            cond_1 = (state[0] + 1) < t_horiz # controllo di essere dentro l'orizzonte temporale alla fine dell'azione di throwing
            cond_2 = state[(n_trays+1) * o + 2] > (state[(n_trays+1) * o + 3] + state[(n_trays+1) * o + 4]) # controllo di avere almeno un oggetto di tipo o sul vassio
            if (cond_1 & cond_2):
                p1_ay[ns] = 100
                # success state
                r1_A_ay[ns] =  -picking_reward  
                s1_A_mtx[ns, :] = state[:]
                s1_A_mtx[ns, 0] = state[0] + 1
                s1_A_mtx[ns, (n_trays+1)*o+2] -= 1
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


def test_mission_once():

    mission =  {
        "tray0": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
        "tray1": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
        "index": 0
    }
    try:
        print(f"test prima missione")
        test_instance = Test(mission)
        time = test_instance.measure_backward_pass_time(mission)
        print(f'tempo prima : {time}')
    except Exception:
        print('ahia!')
        time = "not resolved due to memory error"
        
    mission = {
        "tray0": {"object0": 3, "object1": 0, "object2": 3, "object3": 0, "object4": 0},
        "tray1": {"object0": 0, "object1": 3, "object2": 0, "object3": 0, "object4": 0},
        "index": 1
    }
    try:
        print('test seconda missione')
        test_instance = Test(mission)
        time = test_instance.measure_backward_pass_time(mission)
        print(f'tempo seconda missione : {time}')
    except Exception:
        print(f"ahia!")
        time = "not resolved due to memory error"




def test_mission(mission):
    try:
        print(f"Mission {mission['index']}")
        test_instance = Test(mission)
        time = test_instance.measure_backward_pass_time(mission)
    except Exception:
        print(f"Mission {mission['index']} - Ahia, out of memory!")
        time = "not resolved due to memory error"
    
    return {"Mission": mission, "Elapsed Time": time}

def test_mission_computing_time():
    print('.:::. Starting tests .:::.')

    missions_combinations = []
    #aggiungo tutte le missioni con un numero di oggetti fra 0 ed 1
    for index, (object0_tray0, object1_tray0, object2_tray0, object3_tray0, object4_tray0,
               object0_tray1, object1_tray1, object2_tray1, object3_tray1, object4_tray1) in enumerate(
                itertools.product(range(2), repeat=10), start=1):
        mission = {"tray0":{"object0":object0_tray0,"object1":object1_tray0,"object2":object2_tray0,"object3":object3_tray0,"object4":object4_tray0},
                   "tray1":{"object0":object0_tray1,"object1":object1_tray1,"object2":object2_tray1,"object3":object3_tray1,"object4":object4_tray1},
                   "index": index}
        missions_combinations.append(mission)
    #aggiungo tutte le missioni con un numero di oggetti pari a 5 per un oggetto e 0 per tutti gli altri
    for index, object_index in enumerate(range(5), start=1):
        mission = {
            "tray0": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
            "tray1": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
            "index": index
        }
        mission2 = {
            "tray0": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
            "tray1": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
            "index": index
        }
        mission["tray0"][f"object{object_index}"] = 5
        mission2["tray1"][f"object{object_index}"] = 5
        missions_combinations.append(mission)
        missions_combinations.append(mission2)
    #3 oggetti solo di 1 tipo
    #aggiungo tutte le missioni con un numero di oggetti pari a 3 per un oggetto e 0 per tutti gli altri
    for index, object_index in enumerate(range(5), start=1):
        mission = {
            "tray0": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
            "tray1": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
            "index": index
        }
        mission2 = {
            "tray0": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
            "tray1": {"object0": 0, "object1": 0, "object2": 0, "object3": 0, "object4": 0},
            "index": index
        }
        mission["tray0"][f"object{object_index}"] = 3
        mission2["tray1"][f"object{object_index}"] = 3
        missions_combinations.append(mission)
        missions_combinations.append(mission2)
    #due oggetti con quantità 3 (tutte le combinazioni di coppie) e gli  altri 0
    index = 1
    from itertools import combinations
    objects = ["object0", "object1", "object2", "object3", "object4"]

    for combo in combinations(objects, 2):
        mission = {
            "tray0": {obj: 0 for obj in objects},
            "tray1": {obj: 0 for obj in objects},
            "index": index
        }
        
        for obj in combo:
            mission["tray0"][obj] = 3
            mission["tray1"][obj] = 3

        missions_combinations.append(mission)
        index += 1
        
    #tre oggetti con quantità 3 e gli altri 0
    index = 1
    objects = ["object0", "object1", "object2", "object3", "object4"]

    for combo in combinations(objects, 3):
        mission = {
            "tray0": {obj: 0 for obj in objects},
            "tray1": {obj: 0 for obj in objects},
            "index": index
        }

        for obj in combo:
            mission["tray0"][obj] = 3
            mission["tray1"][obj] = 3

        missions_combinations.append(mission)
        index += 1

    #cinque oggetti con quantità 3 e gli altri 0
    index = 1
    objects = ["object0", "object1", "object2", "object3", "object4"]

    for combo in combinations(objects, 5):
        mission = {
            "tray0": {obj: 0 for obj in objects},
            "tray1": {obj: 0 for obj in objects},
            "index": index
        }

        for obj in combo:
            mission["tray0"][obj] = 3
            mission["tray1"][obj] = 3

        missions_combinations.append(mission)
        index += 1
        
    #sette oggetti con quantità 3 e gli altri 0
    index = 1
    objects = ["object0", "object1", "object2", "object3", "object4"]

    for combo in combinations(objects, 7):
        mission = {
            "tray0": {obj: 0 for obj in objects},
            "tray1": {obj: 0 for obj in objects},
            "index": index
        }

        for obj in combo:
            mission["tray0"][obj] = 3
            mission["tray1"][obj] = 3

        missions_combinations.append(mission)
        index += 1
        
        

    #tutte le missioni con 3 oggetti per ogni tipo ed ogni tray
    mission = {
        "tray0": {"object0": 3, "object1": 3, "object2": 3, "object3": 3, "object4": 3},
        "tray1": {"object0": 3, "object1": 3, "object2": 3, "object3": 3, "object4": 3},
        "index": index
    }
    missions_combinations.append(mission)
    print(f' Totale numero missioni: {len(missions_combinations)}')

    
    results = []
    import multiprocessing
    with multiprocessing.Pool() as pool:
        results = pool.map(test_mission, missions_combinations)

    # Creare un DataFrame con i risultati finali
    df = pd.DataFrame(results)

    # Salvare il DataFrame in un file Excel
    df.to_excel("backward_pass_times.xlsx", index=False)
#%%
if __name__ == '__main__':
    # missione = {"tray0":{"object0":0,"object1":0,"object2":0,"object3":0,"object4":0},
    #         "tray1":{"object0":0,"object1":0,"object2":0,"object3":0,"object4":0}}
    # test = Test(missione)
    # test_mission_computing_time()
    test_mission_once()
    # s0 = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    # t_horizon = test.t_horizon
    # o_max=test.o_max               
    # n_action_nodes=test.n_action_nodes
    # n_objects=test.n_objects
    # n_trays=test.n_trays
    # picking_reward=test.picking_reward
    # placing_reward=test.placing_reward
    # picking_time=test.picking_time
    # throwing_time=test.throwing_time
    # quantities=test.quantities
    # navigation_risk_mtx=test.navigation_risk_mtx
    # picking_prob_mtx=test.picking_prob_mtx
    # throwing_prob_mtx=test.throwing_prob_mtx
    # alpha_risk=test.alpha_risk
    # #%%
    # p1,r1s,s1s,r1f,s1f = _available_actions(s0,t_horizon,o_max,n_action_nodes,n_objects,n_trays,
    #                                 picking_reward,placing_reward,picking_time,throwing_time,
    #                                 quantities,navigation_risk_mtx,picking_prob_mtx,throwing_prob_mtx,alpha_risk)
    
    
    

# %%
