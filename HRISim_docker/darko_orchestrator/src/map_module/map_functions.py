
import numpy as np

from matplotlib import pyplot as plt
from datetime import datetime

def get_costmap_x_y(world_x, world_y, origin_x, origin_y, res):
    costmap_j = int(round((world_x - origin_x) / res))
    costmap_i = int(round((world_y - origin_y) / res))
    return costmap_j, costmap_i

def plot_heatmap(costmap):

    plt.imshow(costmap, cmap='hot', interpolation='nearest')
    current_time = datetime.now()
    plt.savefig(f"/root/output/merged_{current_time.strftime('%Y-%m-%d_%H:%M:%S')}.png")

def merge_costmap(local_costmap, global_costmap):

    res = local_costmap.resolution

    local_origin_x = local_costmap.origin.position.x
    local_origin_y = local_costmap.origin.position.y
    local_max_x = local_origin_x + res * local_costmap.width
    local_max_y = local_origin_y + res * local_costmap.height

    global_origin_x = global_costmap.origin.position.x
    global_origin_y = global_costmap.origin.position.y
    global_max_x = global_origin_x + res * global_costmap.width
    global_max_y = global_origin_y + res * global_costmap.height

    min_x = max(local_origin_x, global_origin_x)
    min_y = max(local_origin_y, global_origin_y)

    max_x = min(local_max_x, global_max_x)
    max_y = min(local_max_y, global_max_y)

    local_min_j, local_min_i = get_costmap_x_y(min_x, min_y, local_origin_x, local_origin_y, res)
    local_max_j, local_max_i = get_costmap_x_y(max_x, max_y, local_origin_x, local_origin_y, res)

    global_min_j, global_min_i = get_costmap_x_y(min_x, min_y, global_origin_x, global_origin_y, res)
    global_max_j, global_max_i = get_costmap_x_y(max_x, max_y, global_origin_x, global_origin_y, res)

    local_data = local_costmap._grid_data[local_min_i:local_max_i, local_min_j:local_max_j]
    global_data = global_costmap._grid_data[global_min_i:global_max_i, global_min_j:global_max_j]

    data = np.maximum.reduce([local_data, global_data])

    final_data = np.copy(global_costmap._grid_data)
    final_data[global_min_i:global_max_i, global_min_j:global_max_j] = data
    return final_data
