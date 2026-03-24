import numpy as np
import json
import matplotlib.pyplot as plt


def get_cost_from_costmap_x_y(j, i, costmap_data):
    return costmap_data[i][j]


def get_world_x_y(costmap_j, costmap_i, origin_x, origin_y, res):

    world_x = costmap_j * res + origin_x
    world_y = costmap_i * res + origin_y
    return world_x, world_y


def get_costmap_x_y(world_x, world_y, origin_x, origin_y, res):

    costmap_j = int(round((world_x - origin_x) / res))
    costmap_i = int(round((world_y - origin_y) / res))
    return costmap_j, costmap_i


def reduce_costmap(x_min, y_min, x_max, y_max, costmap):

    costmap_origin_x = costmap['info']['origin']['position']['x']
    costmap_origin_y = costmap['info']['origin']['position']['y']
    costmap_res = costmap['info']['resolution']

    j_min, i_min = get_costmap_x_y(
        x_min,
        y_min,
        costmap_origin_x,
        costmap_origin_y,
        costmap_res
    )

    j_max, i_max = get_costmap_x_y(
        x_max,
        y_max,
        costmap_origin_x,
        costmap_origin_y,
        costmap_res
    )

    costmap_reduced = {}
    costmap_reduced['data'] = np.array(costmap['data'][i_min:i_max, j_min:j_max])
    costmap_reduced['height'], costmap_reduced['width'] = costmap_reduced['data'].shape
    costmap_reduced['resolution'] = costmap["info"]['resolution']
    costmap_reduced['origin'] = {}
    costmap_reduced['origin']['x'], costmap_reduced['origin']['y'] = get_world_x_y(
        j_min,
        i_min,
        costmap_origin_x,
        costmap_origin_y,
        costmap_res
    )

    return costmap_reduced


def get_costmap_reduced(costmap_subscriber):

    with open("../static_data/reduced_global_map_parameters.json", "r") as file:
        params = json.load(file)

    costmap = {
        "info": {
            "resolution": costmap_subscriber.resolution,
            "width": costmap_subscriber.width,
            "height": costmap_subscriber.height,
            "origin": {
                "position": {
                    "x": costmap_subscriber.origin.position.x,
                    "y": costmap_subscriber.origin.position.y,
                    "z": costmap_subscriber.origin.position.z
                }
            }
        },
        "data": costmap_subscriber._grid_data
    }

    costmap['data'] = np.array(costmap['data']).reshape((costmap["info"]['height'], costmap["info"]['width']))

    x_min = params["x_min"]
    y_min = params["y_min"]
    x_max = params["x_max"]
    y_max = params["y_max"]

    costmap_reduced = reduce_costmap(
        x_min,
        y_min,
        x_max,
        y_max,
        costmap
    )

    return costmap_reduced


def modify_costmap(costmap, x, y, rectangle_height=20, rectangle_width=30, max_value=100):

    half_height = rectangle_height // 2
    half_width = rectangle_width // 2

    map_height, map_width = costmap.shape

    degrade_values = [100, 98, 95, 92, 90]

    for i in range(-half_height, half_height + 1):
        for j in range(-half_width, half_width + 1):

            distance = max(abs(i), abs(j))
            index = int((distance / half_width) * (len(degrade_values) - 1))
            new_value = degrade_values[min(index, len(degrade_values) - 1)]

            if 0 <= y + i < map_height and 0 <= x + j < map_width:
                costmap[y + i, x + j] = new_value

    return costmap


def insert_obstacles(costmap_reduced, location_coordinates):

    for k in location_coordinates.keys():

        j, i = get_costmap_x_y(
            location_coordinates[k]["x"],
            location_coordinates[k]["y"],
            costmap_reduced['origin']['x'],
            costmap_reduced['origin']['y'],
            costmap_reduced['resolution'],
        )

        costmap_reduced['data'] = modify_costmap(costmap_reduced['data'], j, i)

    return costmap_reduced


def plot_costmap(costmap_reduced):
    plt.imshow(costmap_reduced['data'], cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()
