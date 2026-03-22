import pickle
import json
import random
import networkx as nx
import constants as constants

TASK_LIST = {tod.value: [] for tod in constants.TOD}
   
if __name__ == "__main__":  
    GPATH = "/home/hrisim/ros_ws/src/HRISim/peopleflow/peopleflow_manager/res/INB_3floor/graph.pkl"
    with open(GPATH, 'rb') as f:
        G = pickle.load(f)

    random_target = 200
    whereIam = 'T'
    for tod in constants.TOD:
        if tod == constants.TOD.POSTER:
            for s in range(random_target):
                if whereIam == 'T':
                    TASK_LIST[tod.value].append(constants.WP.ROOM1.value)
                    whereIam = 'B'
                else:
                    target = random.choice([constants.WP.CORRIDOR21.value, constants.WP.CORRIDOR23.value, constants.WP.CORRIDOR7.value])
                    TASK_LIST[tod.value].append(target)
                    whereIam = 'T'
                
        elif tod == constants.TOD.BUFFET:            
            for s in range(random_target):
                if whereIam == 'T':
                    TASK_LIST[tod.value].append(constants.WP.CORRIDOR5.value)
                    whereIam = 'B'
                else:
                    TASK_LIST[tod.value].append(constants.WP.CORRIDOR19.value)
                    whereIam = 'T'
        
    
    with open('/home/hrisim/ros_ws/src/HRISim/hrisim_plans/hardcoded/task_list.json', 'w') as f:
        json.dump(TASK_LIST, f)
