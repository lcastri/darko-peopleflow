import json

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from large_graph_generation.costmap_utils import get_cost_from_costmap_x_y, get_world_x_y, get_costmap_x_y
import shapely.geometry as sg
import rospy


def check_distance(n1, n2, distance_threshold):

    n1x, n1y = n1
    n2x, n2y = n2

    return ( (n1x - n2x)**2 + (n1y - n2y)**2 )**0.5 <= distance_threshold

def check_obstacle_collision(edge, obstacle_polygons):

    edge_line = sg.LineString(edge)

    for obstacle_polygon in obstacle_polygons:
        if edge_line.intersects(obstacle_polygon):
            return True

    return False

def is_free_point(costmap, pi, pj, free_x, free_y):

    if free_x == 0 or free_y == 0:
        if costmap['data'][pi][pj] > 0:
            return False
        return True

    for i in range(max(0, pi - free_y), min(costmap['height'], pi + free_y)):
        for j in range(max(0, pj - free_x), min(costmap['width'], pj + free_x)):
            if costmap['data'][i][j] != 0:
                return False

    return True

def get_free_and_obstacles_real_positions(costmap, free_x, free_y):

    free, obst = [], []

    for i in range(costmap['height']):
        for j in range(costmap['width']):

            x, y = get_world_x_y(
                j,
                i,
                costmap['origin']['x'],
                costmap['origin']['y'],
                costmap['resolution']
            )

            # questo punto po esse nodo o po esse ostacolo
            if is_free_point(costmap, i, j, free_x, free_y):
                free.append((x, y))
            else:
                obst.append((x, y))

    return free, obst

def generate_polygons(labels, points, num_obstacle_clusters):

    obstacle_polygons = []
    for i in range(num_obstacle_clusters):

        # Get indices of points in cluster i
        indices = np.where(labels == i)[0]

        if len(indices) < 4:
            # Skip clusters with fewer than four points: choosing min_distance too high some clusters were having less than 4 points, resulting in an error!
            continue

        # Create polygon from points in cluster i
        coords = points[indices]

        xcoords = [coord[0] for coord in coords]
        ycoords = [coord[1] for coord in coords]

        xmin = np.min(xcoords)
        xmax = np.max(xcoords)
        ymin = np.min(ycoords)
        ymax = np.max(ycoords)

        polygon = sg.Polygon([[xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]])

        obstacle_polygons.append(polygon)

    return obstacle_polygons

def promote_action_nodes(nodes, location_coordinates, picking_distance, throwing_distance):

    distances = {}

    fdist = lambda xp, yp, xl, yl: ((xp - xl)**2 + (yp - yl)**2)**.5

    for node_tuple in nodes:

        node = (node_tuple[0], node_tuple[1])
        name = node_tuple[2]

        distances[node] = {}
        distances[node]['name'] = name

        for location_coordinate in location_coordinates.keys():

            distances[node][location_coordinate] = fdist(
                node[0],
                node[1],
                location_coordinates[location_coordinate]['x'],
                location_coordinates[location_coordinate]['y']
            )

    action_nodes = set()

    for location_coordinate in location_coordinates.keys():

        threshold = picking_distance if "box" in location_coordinate else throwing_distance

        [action_nodes.add((node[0], node[1], distances[node]['name'])) for node in distances.keys() if distances[node][location_coordinate] < threshold]


    return {f'n{i}': {'x': node[0], 'y': node[1], 'name': node[2]} for i, node in enumerate(action_nodes)}

def plot_graph(nodes, edges, costmap, action_nodes, obstacle_polygons, large_graph_dict):

    G = nx.Graph()

    # Add all the nodes to the graph with integer labels
    for node in nodes:
        G.add_node(node, label=node, pos=node)

    # Add only the filtered edges to the graph
    for edge in edges:
        G.add_edge(*edge)

    fig, ax = plt.subplots()

    res = costmap["resolution"]
    for i in range(0, costmap["height"], 10):
        for j in range(0, costmap["width"], 10):
            cost = get_cost_from_costmap_x_y(j, i, costmap['data'])

            x, y = get_world_x_y(
                j,
                i,
                costmap['origin']['x'],
                costmap['origin']['y'],
                costmap['resolution']
            )

            if cost > 0:
                ax.fill_between([x, x + res], [y, y], [y + res, y + res], color="k")

    for node in action_nodes.values():
        x, y = node['x'], node['y']
        ax.plot(x, y, "ro")


    for polygon in obstacle_polygons:
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='red', ec='black')  # 'fc' è il colore di riempimento, 'ec' è il colore del bordo


    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, node_size=20, with_labels=False)
    #plt.show()

    fig.savefig('/root/shared/large_graph.png')
    
def get_neighbors(free_positions_discrete, node, num_neighbors):
    """
    Returns the `k` nearest neighbors of `node` in the list `free_positions_discrete`.
    """
    # Create a NearestNeighbors object with n_neighbors=k
    nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(free_positions_discrete)
    # Find the indices of the `k` nearest neighbors of `node` in `free_positions_discrete`
    distances, indices = nbrs.kneighbors([node])

    # Return a list of the `k` nearest neighbors
    return [tuple(free_positions_discrete[i]) for i in indices[0]]

def create_graph(costmap_reduced, location_coordinates):

    with open("../static_data/graph_params.json", "r") as file:
        graph_params = json.load(file)

    # graph params
    free_x = graph_params['free_x']
    free_y = graph_params['free_y']
    num_clusters = graph_params['num_clusters']
    min_samples = graph_params['min_samples']
    dist_threshold = graph_params['dist_threshold']
    num_obstacle_clusters = graph_params['num_obstacle_clusters']
    num_neighbors = graph_params['num_neighbors']
    picking_distance = graph_params['picking_distance']
    throwing_distance = graph_params['throwing_distance']

    free_positions, obstacle_positions = get_free_and_obstacles_real_positions(costmap_reduced, free_x, free_y)

    # # Create a MiniBatchKMeans object with a minimum of min_samples points per batch
    # kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=min_samples)

    # # Fit the MiniBatchKMeans object to the free_positions array
    # kmeans.fit(free_positions)

    # # Get the cluster centers
    # cluster_centers = kmeans.cluster_centers_

    # # Round the cluster center coordinates to integers
    # nodes = [tuple(pt) for pt in cluster_centers]

    wps = rospy.get_param("/peopleflow/wps")

    nodes = []
    nodes_with_name = []

    for name, wp in wps.items():
        if name not in ['parking', 'door_entrance']:
            nodes_with_name.append((wp['x'], wp['y'], name))
            nodes.append((wp['x'], wp['y']))


    if obstacle_positions:

        # Convert obstacle_real_positions to numpy array
        points = np.array(obstacle_positions)

        # Run KMeans to cluster points
        kmeans = KMeans(n_clusters=num_obstacle_clusters).fit(points)
        labels = kmeans.labels_

        # Creation of polygons to represent obstacles
        obstacle_polygons = generate_polygons(labels, points, num_obstacle_clusters)


    # Create a set of all possible edges connecting each node to its num_neighbors nearest neighbors
    edges = set()
    for node in nodes:
        neighbors = get_neighbors(nodes, node, num_neighbors)
        for neighbor in neighbors:
            if node == neighbor:
                continue
            edge = tuple(sorted([node, neighbor]))
            if not check_obstacle_collision(edge, obstacle_polygons) and check_distance(node, neighbor, dist_threshold):
                edges.add(edge)

    # Filter out the longest edge crossing another edge
    to_remove = set()
    edges = list(edges)
    for i in range(len(edges) - 1):
        edge1 = edges[i]
        if edge1 in to_remove:
            continue
        for j in range(i + 1, len(edges)):
            edge2 = edges[j]
            if edge2 in to_remove:
                continue
            if edge1 == edge2:
                continue
            line1 = sg.LineString(edge1)
            line2 = sg.LineString(edge2)
            if line1.crosses(line2):
                if line1.length > line2.length:
                    to_remove.add(edge1)
                else:
                    to_remove.add(edge2)

    for edge in to_remove:
        edges.remove(edge)

    preds = [edge[0] for edge in edges]
    succs = [edge[1] for edge in edges]

    nodes_to_remove = [node for node in nodes if node not in preds and node not in succs]
    nodes_with_name_to_remove = [node for node in nodes_with_name if (node[0], node[1]) not in preds and (node[0], node[1]) not in succs]

    [nodes.remove(node_to_remove) for node_to_remove in nodes_to_remove]
    [nodes_with_name.remove(node_with_name_to_remove) for node_with_name_to_remove in nodes_with_name_to_remove]

    action_nodes = promote_action_nodes(nodes_with_name, location_coordinates, picking_distance, throwing_distance)
    
    return nodes, nodes_with_name, edges, action_nodes, obstacle_polygons

def process_graph(nodes, edges, costmap):

    nodes_dct_xy = {}
    nodes_dct_xy_inverted = {}
    nodes_dct_ij = {}
    nodes_neighbors = {}
    nodes_conversion_dict = {}

    for i, node in enumerate(nodes):

        nodes_dct_xy[i] = (node[0], node[1])
        nodes_dct_xy_inverted[(node[0], node[1])] = i
        nodes_conversion_dict[node[2]] = i

        ji = get_costmap_x_y(
            node[0],
            node[1],
            costmap['origin']['x'],
            costmap['origin']['y'],
            costmap['resolution']
        )

        nodes_dct_ij[i] = (ji[0], ji[1])

        nodes_neighbors[i] = []

    edges_list = []
    for edge in edges:

        n1 = nodes_dct_xy_inverted[edge[0]]
        n2 = nodes_dct_xy_inverted[edge[1]]

        edges_list.append((
            n1,
            n2
        ))

        nodes_neighbors[n1].append(n2)
        nodes_neighbors[n2].append(n1)

    # get costmap indexes associated with edges
    squares_edge_dict = {}
    for idx, e in enumerate(edges_list):

        j0, i0 = nodes_dct_ij[e[0]]
        j1, i1 = nodes_dct_ij[e[1]]

        jmin = int(np.floor(np.min([j0, j1])))
        jmax = int(np.ceil(np.max([j0, j1])))
        imin = int(np.floor(np.min([i0, i1])))
        imax = int(np.ceil(np.max([i0, i1])))

        idx_lst = []

        if jmin == jmax:
            for i in range(imin, imax + 1):
                idx_lst += [(jmin, i)]
        elif imin == imax:
            for j in range(jmin, jmax + 1):
                idx_lst += [(j, imin)]

        else:

            fi = lambda jparam: (i1 - i0) / (j1 - j0) * jparam + (i0 - (i1 - i0) / (j1 - j0) * j0)
            fj = lambda iparam: (j1 - j0) / (i1 - i0) * iparam + (j0 - (j1 - j0) / (i1 - i0) * i0)

            for j in range(jmin, jmax + 1):
                for i in range(imin, imax + 1):
                    if (i <= fi(j) <= i + 1) or (i <= fi(j + 1) <= i + 1) or (j <= fj(i) <= j + 1) or (j <= fj(i + 1) <= j + 1):
                        # if i >= 240 or j >= 180:
                        #     print("ERROR", jmin, imin, jmax, imax, i, j)
                        #     exit(1)
                        idx_lst += [(j, i)]

        squares_edge_dict[idx] = idx_lst

    return {
        "nodes_xy": nodes_dct_xy,
        "nodes_ij": nodes_dct_ij,
        "edges": edges_list,
        "nodes_neighbors": nodes_neighbors,
        "squares_edge_dict": squares_edge_dict,
        "nodes_conversion_dict": nodes_conversion_dict
    }
