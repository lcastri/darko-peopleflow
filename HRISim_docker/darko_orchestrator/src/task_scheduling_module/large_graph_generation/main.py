import json
from large_graph_generation.costmap_utils import get_costmap_reduced, insert_obstacles
from large_graph_generation.graph_utils import create_graph, plot_graph, process_graph

def generate_large_graph(costmap_subscriber):

    with open("../static_data/location_coordinates.json", "r") as file:
        location_coordinates = json.load(file)

    with open("../static_data/graph_params.json", "r") as file:
        graph_params = json.load(file)

    costmap_reduced = get_costmap_reduced(costmap_subscriber)
    # costmap_reduced = insert_obstacles(costmap_reduced, location_coordinates)
    nodes, nodes_with_name, edges, action_nodes, obstacle_polygons = create_graph(costmap_reduced, location_coordinates, graph_params)
    large_graph_dict = process_graph(nodes_with_name, edges, costmap_reduced)

    plot_graph(nodes, edges, costmap_reduced, action_nodes, obstacle_polygons, location_coordinates, graph_params)

    with open("../static_data/large_graph.json", "w") as file:
        json.dump(large_graph_dict, file, indent=4)

    with open("../static_data/action_graph_nodes.json", "w") as file:
        json.dump(action_nodes, file, indent=4)

    with open("/root/shared/large_graph.json", "w") as file:
        json.dump(large_graph_dict, file, indent=4)

    with open("/root/shared/action_graph_nodes.json", "w") as file:
        json.dump(action_nodes, file, indent=4)

