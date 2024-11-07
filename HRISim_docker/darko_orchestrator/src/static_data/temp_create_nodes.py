#%%
import numpy as np
import json
import matplotlib.pyplot as plt


### picking
# {"box0":{"x":3.2,"y":-1.4}, ->
#  "box1":{"x":2.6,"y":-1.4},
#  "box2":{"x":2.0,"y":-1.4},
#  "box3":{"x":1.4,"y":-1.4},
#  "box4":{"x":0.8,"y":-1.4},

### placing
#  "tray0":{"x":2.6,"y":1.4}
#  "tray1":{"x":1.4,"y":1.4}

# X   -1 -> 4   ( 5 ) 
# Y -1.4 -> 1.4 (2.8)

Xmin,Xmax =   -1,  4
Ymin,Ymax = -1.4,1.4

large_graph_dict = {}

d = 0.2
xrange = np.linspace(Xmin,Xmax,int(1+np.round((Xmax-Xmin)/d)))
yrange = np.linspace(Ymin,Ymax,int(1+np.round((Ymax-Ymin)/d)))
n=0
for x in xrange:
    for y in yrange:
        large_graph_dict["n"+str(n)] = {"x":x,"y":y}
        n+=1

with open('large_graph_nodes.json', 'w') as f:
    json.dump(large_graph_dict, f)

#->

slist = ["n135","n180","n225","n270","n315","n194","n284"]
slist += ["n147","n192","n237","n282","n327","n119","n359"]
slist += ["n145","n190","n235","n280","n325"]

action_graph_dict = {}
n=0
for sn in slist:
    action_graph_dict["n"+str(n)] = large_graph_dict[sn]
    n+=1

with open('action_graph_nodes.json', 'w') as f:
    json.dump(action_graph_dict, f)


# # %%

# fig,ax = plt.subplots(figsize=(12,12))
# for on in range(390):
#     n = "n"+str(on)
#     if n in slist:
#         ax.plot(large_graph_dict[n]["x"],large_graph_dict[n]["y"],"ro",markersize=8)
#         ax.text(large_graph_dict[n]["x"],large_graph_dict[n]["y"],on,size=11,ha="center",va="bottom")
#     else:
#         ax.plot(large_graph_dict[n]["x"],large_graph_dict[n]["y"],"ko",markersize=8)
#         ax.text(large_graph_dict[n]["x"],large_graph_dict[n]["y"],on,size=11,ha="center",va="bottom")

# %%
