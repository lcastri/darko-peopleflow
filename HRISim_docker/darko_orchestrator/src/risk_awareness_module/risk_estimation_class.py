#%%

import math
import json
import numpy as np
import numba as nb
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from datetime import datetime


class RiskEstimation:

    def __init__(self, costmap_subscriber, gridmap_subscriber,
                       static_data_path="../static_data", 
                       manipulation_model_path="./manipulation_models"):
        
        # full costmap
        self.costmap_subscriber = costmap_subscriber
        self.gridmap_subscriber = costmap_subscriber # TODO da cambiare
        
        """
        Load static data:
        
        - file action_graph_nodes.json: contains the x,y coordinates of the action nodes and the node name.

        - file large_graph.json: contains 
            - nodes_xy: the x,y coordinates of the nodes
            - nodes_ij: the i,j coordinates of the nodes (according to the costmap)
            - edges: the edges of the graph
            - nodes_neighbors: the neighbors of each node
            - squares_edge_dict: the list of cells of the costmap that compose each edge
            - nodes_conversion_dict: the conversion between node name and node index
        
        - file location_coordinates.json: contains the x,y coordinates of the boxes and trays.

        - file reduced_global_map_parameters.json: contains the parameters to reduce the global costmap.

        - file risk_parameters.json: contains the parameters for the risk estimation.

        - file graph_params.json: contains the parameters for the graph (used for the generation of the graph. In this module 
                                  it is used to get the throwing distance).
       
        - file rbf_picking_parameters_new.json: contains the parameters for the rbf interpolation of the picking model.

        - file rbf_throwing_parameters_new.json: contains the parameters for the rbf interpolation of the throwing model.
          """

        with open(static_data_path + "/action_graph_nodes.json", "r") as json_file:
            self.action_graph_nodes_params = json.load(json_file)
        
        with open(static_data_path + "/large_graph.json", "r") as json_file:
            self.large_graph_params = json.load(json_file, parse_int=int)

        with open(static_data_path + "/location_coordinates.json", "r") as json_file:
            self.location_coordinates_params = json.load(json_file)

        with open(static_data_path + "/reduced_global_map_parameters.json", "r") as json_file:
            self.reduced_global_map_parameters = json.load(json_file)
        
        with open(static_data_path + "/risk_parameters.json", "r") as json_file:
            self.risk_params = json.load(json_file)

        with open(static_data_path + "/graph_params.json", "r") as json_file:
            self.graph_params = json.load(json_file)
        
        # interpolation models for manipulation
        with open(manipulation_model_path + "/rbf_picking_parameters_new.json", "r") as json_file:
            self.rbf_pick_param = json.load(json_file)
        
        with open(manipulation_model_path + "/rbf_throwing_parameters_new.json", "r") as json_file:
            self.rbf_throw_param = json.load(json_file)

        """GRAPH NAMES CONVERSION, COSTAMP REDUCTION AND PARAMETERS DEFINITION"""
        # conversion between node name "n0", "n1", ... and node index "0", "1", ...
        self.action_graph_conversion_dict = {n: int(n[1:]) for n in self.action_graph_nodes_params.keys()}
        self.action_idx_name = self.get_node_action_idx_from_large_graph()

        self.reduced_map = Global_costamap_reduction(self.costmap_subscriber,self.reduced_global_map_parameters)
        self.reduced_static_map = Global_costamap_reduction(self.gridmap_subscriber,self.reduced_global_map_parameters)

        self.update_parameters = True
        self.v_max = self.risk_params["v_max"]
        self.alpha = self.risk_params["alpha"]
        self.plot = False

        """ FOR THE MANIPULATION RISK ESTIMATION """
        # initialize the rbf interpolation models
        self.rbf_interp_picking = RBFInterpolation(self.rbf_pick_param["sigma"], self.rbf_pick_param["centers"], self.rbf_pick_param["weights"], self.rbf_pick_param["alpha"],)
        self.rbf_interp_throwing = RBFInterpolation(self.rbf_throw_param["sigma"], self.rbf_throw_param["centers"], self.rbf_throw_param["weights"], self.rbf_throw_param["alpha"])
        
        # to include the risk along the trajectories when computing the throwing/picking risk 
        # define the location of the boxes and trays (list of tuples (x,y) of trays and boxes)
        self.boxes_loc, self.trays_loc = self.define_boxes_trays_loc()
        # define the internal points of the area that contains the trays/boxes
        self.trays_area_points = self.define_location_area(self.trays_loc, self.risk_params["loc_square_edge"])
        self.boxes_area_points = self.define_location_area(self.boxes_loc, self.risk_params["loc_square_edge"])
        # define the internal points of the trajectories between node_action and trays/boxes
        self.tray_trajectory_points = self.define_trajectory_area(self.trays_loc, self.trays_area_points, self.risk_params["trajectory_polygon_width"],self.risk_params["node_side_length"]) 
        self.boxes_trajectory_points = self.define_trajectory_area(self.boxes_loc, self.boxes_area_points, self.risk_params["trajectory_polygon_width"],self.risk_params["node_side_length"]) 
        
        # compute the distances between the action nodes and the boxes and trays for the rbf
        self.distances_dict = {}   
        for node in self.action_graph_nodes_params: 
            x, y = self.action_graph_nodes_params[node]["x"], self.action_graph_nodes_params[node]["y"] # node location
            d_boxes = self.get_distances_from_locations(x,y, self.boxes_loc)
            d_trays = self.get_distances_from_locations(x,y, self.trays_loc)
            self.distances_dict[node] = (d_boxes, d_trays)

        self.num_boxes = len(self.boxes_loc)
        self.num_trays = len(self.trays_loc)

        """MOCK DELLA PARTE DI LUCA C"""
        
        self.t_list = self.risk_params["t_list"]
        self.dynamic_costmap_weight ={t: np.exp(-self.risk_params["dynamic_costmap_weight_k"] * t) for t in self.t_list}
       
        n_row = len(self.large_graph_params['nodes_xy'])
        n_col = len(self.t_list)
        self.prediction_risk_matrix = 0 * np.ones((n_row, n_col), dtype=np.float64)
 
        self.prediction_risk_matrix_names = list(self.large_graph_params['nodes_conversion_dict'].keys())

        self.predictions_costmaps_dict = {t: np.zeros_like(self.reduced_map.data, dtype=float) for t in self.t_list}
        self.merged_costmaps_dict = {t: np.zeros_like(self.reduced_map.data, dtype=int) for t in self.t_list}

        self.gaussian_sigma = 15.0
        self.r = 25

        _ = self.get_risk_estimations()
        

# RISK ESTIMATION

    def get_node_action_idx_from_large_graph(self):
        nodes_action_idx_map = {}
        nodes_action = self.action_graph_nodes_params
        graph_nodes_xy = self.large_graph_params["nodes_xy"]
        for n0 in nodes_action:
            for n in graph_nodes_xy:
                if (graph_nodes_xy[n][0]==nodes_action[n0]["x"] and graph_nodes_xy[n][1]==nodes_action[n0]["y"]):
                    nodes_action_idx_map[int(n)] = self.action_graph_conversion_dict[n0]
        return nodes_action_idx_map

    def get_risk_estimations(self):

        self.generate_prediction_risk_costmaps()

        self.merge_costmaps()

        navigation_risk_mtx = self.get_navigation_risk()
        pick_risk_mtx,throw_risk_mtx = self.get_manipulation_risk()

        return navigation_risk_mtx,pick_risk_mtx,throw_risk_mtx

    def get_navigation_risk(self):
        n_action_nodes = len(self.action_graph_nodes_params)
        navigation_risk_mtx = np.zeros((n_action_nodes,n_action_nodes, 4*len(self.t_list)), dtype=np.float64)
        # for each time step, compute the navigation risk matrix
        for t_idx in range(len(self.t_list)):
            navigation_risk_mtx[:,:,4*t_idx:4*(t_idx+1)] = get_navigation_risk_mtx(self.merged_costmaps_dict[self.t_list[t_idx]], self.reduced_map, self.large_graph_params, self.action_idx_name, self.v_max, 
                                                       self.action_graph_nodes_params, self.action_graph_conversion_dict, self.risk_params)
        
        return navigation_risk_mtx

    def get_manipulation_risk(self):
        n_action_nodes = len(self.action_graph_nodes_params)
        pick_mtx = np.zeros((n_action_nodes, self.num_boxes*len(self.t_list)),dtype=np.int64)
        throw_mtx = np.zeros((n_action_nodes, self.num_trays*len(self.t_list) ),dtype=np.int64)

        for t_idx in range(len(self.t_list)):
            if not self.plot:
                
                pick_mtx_t, throw_mtx_t = get_manipulation_risk(self.action_graph_nodes_params, self.action_graph_conversion_dict,  
                                                        self.num_boxes, self.num_trays, self.distances_dict,
                                                        self.rbf_interp_picking, self.rbf_interp_throwing, self.merged_costmaps_dict[self.t_list[t_idx]], 
                                                        self.tray_trajectory_points, self.boxes_trajectory_points,self.risk_params)

                pick_mtx[:,self.num_boxes*t_idx:self.num_boxes*(t_idx+1)] = pick_mtx_t
                throw_mtx[:,self.num_trays*t_idx:self.num_trays*(t_idx+1)] = throw_mtx_t
            else:
                pick_mtx_t, throw_mtx_t = get_manipulation_risk_with_plot(self.action_graph_nodes_params, self.action_graph_conversion_dict, 
                                                        self.num_boxes, self.trays_loc, self.distances_dict,
                                                        self.rbf_interp_picking, self.rbf_interp_throwing,self.merged_costmaps_dict[self.t_list[t_idx]],self.reduced_map, 
                                                        self.tray_trajectory_points, self.boxes_area_points, self.trays_area_points, self.risk_params)
                pick_mtx[:,self.num_boxes*t_idx:self.num_boxes*(t_idx+1)] = pick_mtx_t
                throw_mtx[:,self.num_trays*t_idx:self.num_trays*(t_idx+1)] = throw_mtx_t
        
        return pick_mtx, throw_mtx

# COSTMAPS REDUCTION, MERGING AND DEFINITION

    def update_reduce_map(self):

        self.reduced_map.data = np.copy(
            self.costmap_subscriber._grid_data[
                self.reduced_map.y_min:self.reduced_map.y_max,
                self.reduced_map.x_min:self.reduced_map.x_max
            ]
        )

    def generate_prediction_risk_costmaps(self):
        """
        Generate the prediction risk costmaps for each time step in the dictionary predictions_costmaps_dict.
        """
        # the prediction costmap is a costmap where the risk values from prediction have been spreaed with a Gaussian Kernel
        self.predictions_costmaps_dict = {t:np.zeros_like(self.reduced_map.data, dtype=float) for t in self.t_list}
        for t_idx in range(len(self.t_list)):
            for i in range(len(self.prediction_risk_matrix)):
                risk_value = self.prediction_risk_matrix[i, t_idx]
                node_idx = self.large_graph_params['nodes_conversion_dict'][self.prediction_risk_matrix_names[i]]
                j,i = self.large_graph_params['nodes_ij'][str(node_idx)]
                # Propagate the risk value to the surrounding cells with a Gaussian kernel
                for di in range(-self.r, self.r + 1):
                    for dj in range(-self.r, self.r + 1):
                        ni, nj = i + di, j + dj
                        if self.reduced_map.is_in_gridmap(ni, nj):
                            distance = np.sqrt(di**2 + dj**2)
                            if distance <= self.r:
                                kernel_value = np.exp(-distance**2 / (2 * self.gaussian_sigma**2))
                                # it can happen to combine the risk value spreaded from different near nodes
                                self.predictions_costmaps_dict[self.t_list[t_idx]][ni, nj] += risk_value * kernel_value
            self.predictions_costmaps_dict[self.t_list[t_idx]] = np.minimum(100, self.predictions_costmaps_dict[self.t_list[t_idx]])

    def merge_costmaps(self):
        """
        Merge of the dynamic costmap, static costmap and risk predictions. 
        Updates the merged costmap for each time step in the dictionary merged_costmaps_dict.
        """
        self.update_reduce_map()

        dynamic_costmap = self.reduced_map.data.copy()
        static_costmap = self.reduced_static_map.data.copy()
        assert dynamic_costmap.shape == static_costmap.shape, "The dynamic and static costmaps have different shapes"
        
        dynamic_costmap_dict = {}
        for t in self.t_list:
            dynamic_costmap_dict[t] = dynamic_costmap.copy()
            

        # The dynamic costmap is weighted by a coefficient decreasing over time
        for t in self.t_list:
            # The merged costmap is the maximum between the dynamic costmap, the static costmap and the prediction costmap   
            merge_predictions_dynamic = np.maximum(self.predictions_costmaps_dict[t], self.dynamic_costmap_weight[t]*dynamic_costmap_dict[t])
            self.merged_costmaps_dict[t]  = np.maximum(merge_predictions_dynamic, static_costmap)

# AREAS AND TRAJECTORIES DEFINITION
    def define_boxes_trays_loc(self):
    # return a list with the tuples (x,y) of the boxes and a list with the (x,y) of the trays
        boxes_loc = []
        trays_loc = []
        for key, value in self.location_coordinates_params.items():
            if key.startswith('box'):
                boxes_loc.append((value['x'], value['y']))
            elif key.startswith('tray'):
                trays_loc.append((value['x'], value['y'])) 
        return boxes_loc, trays_loc

    def define_location_area(self, loc_list, square_edge):
        # takes in input a list of (x,y) of trays or boxes and returns the internal points of the area containing the trays/boxes
        total_internal_points = []
        for t in loc_list:
            i, j = self.reduced_map.get_costmap_x_y(t[0], t[1])
            # Calcola metà della lunghezza del lato
            half_edge = square_edge / 2
            half_edge2 = square_edge / 2 + 1
            # Calcola i vertici del quadrato
            top_left = (max(int(i - half_edge2), 0), max(int(j - half_edge), 0))
            # top_right = (max(int(i - half_edge2), 0), min(int(j + half_edge), self.reduced_map.width- 1))
            # bottom_left = (min(int(i + half_edge2), self.reduced_map.lenght - 1), max(int(j - half_edge), 0))
            bottom_right = (min(int(i + half_edge2), self.reduced_map.lenght  - 1), min(int(j + half_edge), self.reduced_map.width - 1))
            # corners = np.array([top_left, top_right, bottom_right, bottom_left])
            # Trova tutti i punti interni
            internal_points =  [(x, y) for x in range(top_left[0], bottom_right[0] + 1) for y in range(top_left[1], bottom_right[1] + 1)]
            total_internal_points = total_internal_points + internal_points
        return total_internal_points

    def define_trajectory_area(self, loc_list, loc_area_points, trajectory_polygon_width, node_side_length): 
        # define for each node, for each tray/box, the internal points of the trajectory that connects the node to the target (tray/box).
        # the trajectory area is given by the sum of two shapes: a square centered in the node and a trapezoid connecting the two points (node and target)
        # I get two dictionaries with key the pair of indices (node,box/tray) and value the list of points internal to the trajectory.
        traj_internal_points_dict = {}
        for node in self.action_graph_nodes_params:
            x, y = self.action_graph_nodes_params[node]["x"], self.action_graph_nodes_params[node]["y"] # node location
            n = self.action_graph_conversion_dict[node]
            for idx in range(0, len(loc_list)): 
                x_l, y_l = loc_list[idx][0], loc_list[idx][1]
                distance = euclidean_distance(x, y, x_l, y_l) # Calcola la distanza tra il nodo e il box/tray
                if distance <= self.graph_params["throwing_distance"]:
                    internal_points = self.define_trajectory_internal_points(x, y, x_l, y_l, loc_area_points, trajectory_polygon_width, node_side_length)
                    traj_internal_points_dict[(n, idx)] = internal_points
                else:
                    traj_internal_points_dict[(n, idx)] = []
        return traj_internal_points_dict

    def define_trajectory_internal_points(self, x_n, y_n, x_o, y_o, obj_internal_points, trajectory_polygon_width, node_side_length):
        # define the internal points of the trajectory area that connects the node to the target (tray/box).
        i_o, j_o = self.reduced_map.get_costmap_x_y(x_o, y_o)
        i_n, j_n = self.reduced_map.get_costmap_x_y(x_n, y_n)
        mid_point_1 = np.array([i_o,j_o])
        mid_point_2 = np.array([i_n,j_n])

        # Definisco il trapezio che congiunge il nodo al tray/box
        # Calcola il vettore direzione tra i punti medi e il vettore perpendicolare
        direction_vector = np.array(mid_point_2) - np.array(mid_point_1)
        v = direction_vector / np.linalg.norm(direction_vector)
        perp_vector = (np.array([-v[1], v[0]]))* (trajectory_polygon_width / 2)
        perp_vector2 = (np.array([-v[1], v[0]]))* (trajectory_polygon_width / 1.2)
        # Calcola i vertici del poligono (trapezio)
        top_left = (mid_point_1 + perp_vector2).astype(int)
        top_right = (mid_point_1 - perp_vector2).astype(int)
        bottom_left = (mid_point_2 + perp_vector).astype(int)
        bottom_right = (mid_point_2 - perp_vector).astype(int)
        corners = np.array([top_left, top_right, bottom_right, bottom_left])
        polygon = Polygon(corners)
        # Calcolo i punti dell'area intorno al poligono (poligono allineato agli assi)
        i_coord = np.array([top_left[0], top_right[0], bottom_right[0], bottom_left[0]])
        j_coord = np.array([top_left[1], top_right[1], bottom_right[1], bottom_left[1]])
        min_i, min_j  = np.min(i_coord), np.min(j_coord)
        max_i, max_j = np.max(i_coord), np.max(j_coord)
        # Verifico quali di questi punti è interno al trapezio
        internal_points = []
        for i in range(min_i, max_i+1):
            for j in range(min_j, max_j+1):
                point = Point(i, j)
                if polygon.contains(point):
                    internal_points.append((i, j)) # punti interni al trapezio

        # Definisco un quadrato centrato nel nodo da cui fare l'azione    
        half_side = node_side_length / 2
        # Calcola i limiti del quadrato
        i_min = int(i_n - half_side)
        i_max = int(i_n + half_side)
        j_min = int(j_n - half_side)
        j_max = int(j_n + half_side)
        # Itera sui punti all'interno dei limiti
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                internal_points.append((i,j)) # aggiungo i punti interni al quadrato a quelli punti interni al trapezio     
        
        # Dalla lista di punti interni alla traiettoria (trapezio + quadrato) sottraggo i punti in overlapping con l'obiettivo box/tray
        traj_internal_points_set = set(internal_points) 
        obj_internal_points_set = set(obj_internal_points) # punti interni all'area che circonda boxes o trays
        trajectory_without_obj= list(traj_internal_points_set - obj_internal_points_set)

        return trajectory_without_obj

    def get_distances_from_locations(self, x,y,loc_):

        distances = np.array([]) 
        for l in loc_: 
            distances_0 = np.array(euclidean_distance(l[0],l[1],x,y)).reshape(-1)
            distances = np.concatenate((distances, distances_0))
        distances = distances.reshape(-1,1)
        return distances

# PLOTS 
    def plot_costmap(self, data):
        plt.figure(figsize=(6, 6))
        plt.imshow(data, cmap='gray', origin='lower')
        plt.colorbar(label='Cost')
        plt.title('ROS Costmap')
        plt.savefig("/root/shared/costmp.png")

    def plot_costmaps(self, costmaps, t_values, alpha_values, k):
        if len(costmaps) != 4 or len(t_values) != 4 or len(alpha_values) != 4:
            raise ValueError("Le liste devono contenere esattamente 4 elementi.")

        fig, axes = plt.subplots(1, 4, figsize=(40, 10))  # Layout orizzontale ottimizzato
        fig.suptitle(f'k = {k}', fontsize=32, fontweight='bold')  # Titolo più grande

        # Normalizziamo i colori per avere una scala comune
        vmin = min(cm.min() for cm in costmaps)
        vmax = max(cm.max() for cm in costmaps)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        im = None  # Variabile per l'ultima immagine
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(costmaps[i], cmap='viridis', norm=norm, origin='upper')
            ax.set_title(f't = {t_values[i]}\nα = {alpha_values[i]:.2f}', fontsize=28, fontweight='bold')
            ax.title.set_position([0.5, 1.05])  # Sposta il titolo in alto
            ax.axis('off')  # Nasconde gli assi

        # Aggiungiamo una colorbar accanto all'ultima costmap
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=24)  # Ingrandisce il font della colorbar

        # Riduciamo gli spazi tra i subplot per ottimizzare la disposizione
        plt.subplots_adjust(left=0.05, right=0.90, top=0.85, bottom=0.15, wspace=0.15)

        plt.show()

# CONTINUOUS LEARNING
    def gradiente(self, t_real, t_prev, velocity):
        grad = -(t_prev-t_real)*velocity/t_real
        return grad

    def update_velocity(self, t_real, t_prev):
            new_v = self.v_max - self.alpha*(self.gradiente(t_real, t_prev, self.v_max ))
            self.v_max = new_v


class Global_costamap_reduction:

    def __init__(self,costmap_subscriber,reduced_global_map_parameters):
        
        self.x_min = reduced_global_map_parameters['x_min']
        self.x_max = reduced_global_map_parameters['x_max']
        self.y_min = reduced_global_map_parameters['y_min']
        self.y_max = reduced_global_map_parameters['y_max']

        self.x_min, self.y_min = costmap_subscriber.get_costmap_x_y(self.x_min, self.y_min)
        self.x_max, self.y_max = costmap_subscriber.get_costmap_x_y(self.x_max, self.y_max)

        self.data = np.copy(costmap_subscriber._grid_data[self.y_min:self.y_max,self.x_min:self.x_max])

        self.origin_x,self.origin_y  = costmap_subscriber.get_world_x_y(self.x_min,self.y_min) 
        self.width,self.lenght = self.data.shape
        self.resolution = costmap_subscriber.resolution
    
    def get_world_x_y(self, costmap_x, costmap_y):
        world_x = costmap_x * self.resolution + self.origin_x
        world_y = costmap_y * self.resolution + self.origin_y
        return world_x, world_y

    def get_costmap_x_y(self, world_x, world_y):
        costmap_x = int(round((world_x - self.origin_x)/self.resolution))
        costmap_y = int(round((world_y - self.origin_y)/self.resolution))
        return costmap_x, costmap_y

    def get_cost_from_costmap_x_y(self, x, y):
        return self.data[y][x]

    def get_cost_from_world_x_y(self, x, y):
        cx, cy = self.get_costmap_x_y(self,x, y)
        return self.get_cost_from_costmap_x_y(self,cx, cy)
    
    def is_in_gridmap(self, x, y):
        if -1 < x < self.width and -1 < y < self.lenght:
            return True
        else:
            return False

class RBFInterpolation:

    def __init__(self, sigma, centers, weights, alpha):
        self.sigma = sigma
        self.centers = centers
        self.weights = weights
        self.alpha = alpha

   # generate the model
    def radial_basis_function(self, x, centers, sigma):
        return np.exp(-np.square(x - centers) / (2 * sigma**2))

    # update the model
    def gradient(self, y_r, y_p, phi_xi):
        grad = -2 *(y_r - y_p)*phi_xi
        return grad

    def update_weights(self, x_r, y_r):
        centers = np.array(self.centers)
        phi_xi = self.radial_basis_function(x_r, centers, self.sigma) 
        y_p = np.dot(phi_xi, self.weights) # prediction
        gradient = self.gradient(y_r, y_p, phi_xi)
        delta_w = self.alpha*gradient
        self.weights = self.weights - delta_w

    def predict(self, X, scaling_factor):
        num_centers = len(self.centers)
        phi_i = np.zeros((num_centers,len(X)))
        for i in range(num_centers):
            '''NEL CASO IN CUI VARIA LA SOGLIA E BISOGNA USARE LO SCALING FACTOR, RICORDATI CHE QUELLA CON CUI SONO STATI TRAINATI I MODELLI E' 1.5'''
            phi_i[i,:] = [self.radial_basis_function(x / scaling_factor, self.centers[i],self.sigma)[0] for x in X]
        y_pred = [np.sum(phi_i[:,j]*self.weights) for j in range(len(X))]
        # print("y_pred", y_pred)
        y_pred_ay = np.array(y_pred).reshape(-1,)
        y_pred_ay = np.round(y_pred_ay*100).astype(int)
        y_pred_ay = np.clip(y_pred_ay, 0, None) # set negative values to zero
        y_pred_ay = [min(x, 100) for x in y_pred_ay] # set values greater than 100 to 100
        # print("y_pred_ay", y_pred_ay)
        return y_pred_ay
    

# -------- manipulation ---------------------------------------------------------

def euclidean_distance(x1,y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def get_corrected_cost(cost):
    if cost < 50:
        return 0
    if  cost < 85:
        return cost/3
    return 100  

def get_manipulation_risk(action_graph_nodes, action_graph_conversion_dict,  n_boxes, n_trays, distances_dict,
                          rbf_interp_picking, rbf_interp_throwing, 
                          merged_costmap, trays_trajectory_internal_points,boxes_trajectory_internal_points, risk_params):
    
    n_nodes_actions = len(action_graph_nodes)
    # evaluate picking and throwing risk for every action node
    pick_mtx = np.zeros((n_nodes_actions, n_boxes),dtype=np.int64)
    throw_mtx = np.zeros((n_nodes_actions, n_trays),dtype=np.int64)
    for node in action_graph_nodes: 
        node_idx = action_graph_conversion_dict[node]
        d_boxes = distances_dict[node][0]
        d_trays = distances_dict[node][1]

        '''NB: i modelli che calcolano la probabilità di successo nei casi di picking e throwing sono stati allenati considerando una distanza di 1.5 
        --> oltre 1.5 la probabilità di successo è 0. Quando le distanze "caratteristiche" cambiano (e.g., un robot può fare picking da 8 metri), non serve ri-allenare 
        i modelli ma è sufficiente utilizzare lo scaling factor presente nei risk parameters'''

        # picking and throwing risk prediction by rbf interpolation
        pick_risk = rbf_interp_picking.predict(d_boxes, risk_params["rbf_pick_scaling_factor"])
        throw_risk = rbf_interp_throwing.predict(d_trays, risk_params["rbf_throw_scaling_factor"])

        # print("pick_risk, throw_risk", pick_risk, throw_risk)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++")
        trays_trajectories_risk = get_manipulation_trajectory_risk(node_idx, n_trays, trays_trajectory_internal_points, merged_costmap)
        boxes_trajectories_risk = get_manipulation_trajectory_risk(node_idx, n_boxes, boxes_trajectory_internal_points, merged_costmap)

        for idx in range(0, n_trays): # se il rischio di fare throwing dal node al tray idx è alto, dividi per tre la prob di successo del lancio
            if trays_trajectories_risk[idx] > 40: 
                throw_risk[idx] = throw_risk[idx]/3
        for idx in range(0, n_boxes): # se il rischio di fare picking dal node alla box idx è alto, dividi per tre la prob di successo del picking
            if boxes_trajectories_risk[idx] > 40: 
                pick_risk[idx] = pick_risk[idx]/3
        # save risk predictions for current node
        pick_mtx[node_idx] = pick_risk
        throw_mtx[node_idx] = throw_risk
    return pick_mtx, throw_mtx

def get_manipulation_risk_with_plot(action_graph_nodes, action_graph_conversion_dict, 
                                    n_boxes, trays_loc, distances_dict,
                                    rbf_interp_picking, rbf_interp_throwing, 
                                    merged_costmap, costmap, 
                                    trays_trajectory_internal_points, boxes_internal_points, trays_internal_points, risk_params):
    n_nodes_actions = len(action_graph_nodes)
    n_trays = len(trays_loc)
    # evaluate picking and throwing risk for every action node
    pick_mtx = np.zeros((n_nodes_actions, n_boxes),dtype=np.int64)
    throw_mtx = np.zeros((n_nodes_actions, n_trays),dtype=np.int64)
    for node in action_graph_nodes: 
        node_idx = action_graph_conversion_dict[node]
        d_boxes = distances_dict[node][0]
        d_trays = distances_dict[node][1]

        '''NB: i modelli che calcolano la probabilità di successo nei casi di picking e throwing sono stati allenati considerando una distanza di 1.5 
        --> oltre 1.5 la probabilità di successo è 0. Quando le distanze "caratteristiche" cambiano (e.g., un robot può fare picking da 8 metri), non serve ri-allenare 
        i modelli ma è sufficiente utilizzare lo scaling factor presente nei risk parameters'''

        # picking and throwing risk prediction by rbf interpolation
        pick_risk = rbf_interp_picking.predict(d_boxes, risk_params["rbf_pick_scaling_factor"])
        throw_risk = rbf_interp_throwing.predict(d_trays, risk_params["rbf_throw_scaling_factor"])

        # print("pick_risk, throw_risk", pick_risk, throw_risk)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++")

        x, y = action_graph_nodes[node]["x"], action_graph_nodes[node]["y"] # node location
        trays_trajectories_risk = get_manipulation_trajectory_risk_with_plot(node_idx, n_trays, trays_trajectory_internal_points, merged_costmap, costmap, x, y, boxes_internal_points, trays_internal_points, trays_loc)

        for idx in range(0, n_trays): # se il rischio di fare throwing dal node al tray idx è alto, dividi per tre la prob di successo del lancio
            if trays_trajectories_risk[idx] > 40: 
                throw_risk[idx] = throw_risk[idx]/3

        pick_mtx[node_idx] = pick_risk
        throw_mtx[node_idx] = throw_risk
    return pick_mtx, throw_mtx

def get_manipulation_trajectory_risk(node, n_loc, trajectory_internal_points, merged_costmap):
    # risk of doing a manipulation action from a node to a tray/box considering the trajectory area
    # return a list with the risks of manipulation action for each box/tray from the given node
    trajectories_risk = []
    for idx in range(0, n_loc):
        internal_points = trajectory_internal_points[(node, idx)]
        # evaluate the costmap values in the trajectory area
        trajectory_area_values = []
        for (j,i) in internal_points:
            try:
                # j,i = costmap.get_costmap_x_y(x,y)
                cost = merged_costmap[i,j]
                cost_corr = get_corrected_cost(cost)
                trajectory_area_values.append(cost_corr)
            except IndexError:
                # Ignora l'errore e continua con il prossimo punto
                continue
        if trajectory_area_values:
            max_value = np.max(trajectory_area_values)
        else:
            max_value = 1
        trajectories_risk.append(max_value)
    return trajectories_risk

# per debug per plottare le traiettorie
def get_manipulation_trajectory_risk_with_plot(node, n_loc, trajectory_internal_points,merged_costmap, costmap,
                                      node_x, node_y, box_internal_points, trays_internal_points, loc_):
    trajectories_risk = []
    for idx in range(0, n_loc):
        internal_points = trajectory_internal_points[(node, idx)]
        # evaluate the costmap values in the trajectory area
        trajectory_area_values = []
        for (x,y) in internal_points:
            try:
                j,i = costmap.get_costmap_x_y(x,y)
                cost = merged_costmap[i,j]
                cost_corr = get_corrected_cost(cost)
                trajectory_area_values.append(cost_corr)
            except IndexError:
                # Ignora l'errore e continua con il prossimo punto
                continue
        if trajectory_area_values:
            max_value = np.max(trajectory_area_values)
        else:
            max_value = 1
        if internal_points:
            plot_manipulation_area(node_x, node_y, loc_[idx][0],loc_[idx][1], merged_costmap,costmap, box_internal_points, trays_internal_points, internal_points)
        trajectories_risk.append(max_value)
    return trajectories_risk

def plot_manipulation_area(node_x, node_y, manip_x, manip_y, merged_costmap, costmap, box_internal_points, trays_internal_points, traj_internal_points):
    # Disegno della traiettoria sulla costmap
    i,j = costmap.get_costmap_x_y(node_x,node_y)
    it,jt = costmap.get_costmap_x_y(manip_x,manip_y)
    fig, ax = plt.subplots()
    ax.imshow(merged_costmap.data, cmap='gray_r', interpolation='nearest')

    # disegno box, trays e traiettorie
    for idx,point in enumerate(traj_internal_points):
        if idx%5 == 0:
            plt.scatter(point[0], point[1], color='orange', s = 1)
    for idx,point in enumerate(box_internal_points):
        if idx%5 == 0:
            plt.scatter(point[0], point[1], color='purple', s = 1)
    for idx,point in enumerate(trays_internal_points):
        if idx%5 == 0:
            plt.scatter(point[0], point[1], color='green', s = 1)
    current_time = datetime.now()

    # disegno i punti di partenza (nodo di lancio e nodo di placing)
    plt.scatter([it], [jt], color='lightgreen', s = 5)
    plt.scatter([i], [j], color='red', s = 5)
    # plt.savefig(f"/root/shared/traj_{current_time.strftime('%Y-%m-%d_%H:%M:%S')}.png")
    plt.savefig(f"traj_{current_time.strftime('%Y-%m-%d_%H%M%S')}.png")



# -------- navigation -----------------------------------------------------------

@nb.njit
def get_corrected_riskmap(c0):
    Kj = c0.shape[0]
    Ki = c0.shape[1]
    # riskmap = c0
    riskmap = c0.copy()
    riskmap = riskmap.transpose()
    count = 0
    for i in range(Ki):
        for j in range(Kj):
            if (riskmap[i,j] < 50):
                riskmap[i,j] = 0
            elif (riskmap[i,j] >= 50) & (riskmap[i,j] < 90):
                riskmap[i,j] = riskmap[i,j]/3
            else:
                riskmap[i,j] = 100
                count += 1
    return riskmap

def get_rt_edges_new(riskmap,edges,nodes_dct,squares_edge_dict,resolution,vmax):
    rt_edges = {}
    for ie in range(len(edges)):
        e = edges[ie]
        x0,y0 = nodes_dct[e[0]]
        x1,y1 = nodes_dct[e[1]]
        dst = euclidean_distance(x0,y0,x1,y1)*resolution
        squares_edge = squares_edge_dict[ie]

        lst = []
        for idx in [tuple(element) for element in squares_edge]:
            try:
                lst.append(riskmap[idx])
            except IndexError as ie:
                pass
                # print(ie, idx)

        # lst = [riskmap[idx] for idx in [tuple(element) for element in squares_edge]]

        rmax,ravg = np.max(lst),np.mean(lst)
        # print(f'rmax: {rmax}')
        # print(f'ravg: {ravg}')
        dens = 0
        for valore in lst:
             if valore != 0:
                dens += 1
        dens_norm = (dens/len(lst))*100
        vreal = vmax*(1-9/1000*ravg)
        dt = dst/vreal
        # rr = 0.6*rmax+0.4*ravg
        rr = 0.5*rmax+0.4*ravg + 0.1*dens_norm
        rt_edges[e] = [dt,rr]
    return rt_edges

def get_NRT_arrays(nodes_dct,nodes_neighbors,edges,rt_edges):
    Nay=np.zeros((len(nodes_dct.keys()),8),dtype=np.int64) # neighbouring nodes
    Ray=np.zeros((len(nodes_dct.keys()),8)) # risk between nodes
    Tay=np.zeros((len(nodes_dct.keys()),8)) # time between nodes

    for i0 in nodes_dct.keys():
        nn=len(nodes_neighbors[i0])
        for j in range(8):
            if (j<nn):
                i1=nodes_neighbors[i0][j]
                if (i0,i1) in edges:
                    t,r=rt_edges[(i0,i1)]
                else:
                    t,r=rt_edges[(i1,i0)]
                Nay[i0,j]=i1
                Tay[i0,j]=t
                Ray[i0,j]=r
            else:
                Nay[i0,j]=-1
                Tay[i0,j]=-1
                Ray[i0,j]=-1
    return Nay,Tay,Ray

def get_PLI_arrays(nodes_action):
    n_nodes_action = len(nodes_action)
    links = []
    for i in range(n_nodes_action-1):
        for j in range(i+1,n_nodes_action):
            links += [(nodes_action[i],nodes_action[j])]
    Lay,Iay=np.zeros((len(links),2),dtype=np.int64),np.zeros((len(links),2)) # Lay:couples of nodes_action; Iay: time and risk for each couple
    Pay=nb.typed.Dict.empty(key_type=nb.core.types.int64, value_type=nb.core.types.int64[:])
    for i in range(len(links)):
        Lay[i,0]=links[i][0]
        Lay[i,1]=links[i][1]
    return Pay,Lay,Iay

@nb.njit
def get_times_and_risks_new(V,Pay,Lay,Iay,steps,Nay,Tay,Ray,alpha_t,alpha_r,q,n_nodes, nodes_action):
    num_iter = 0
    num_iter_end = 0
    for j in range(0, len(nodes_action)-1):
        n_end   = nodes_action[j]
        V = _backward_pass_nb(V,steps,n_nodes,n_end,Nay,Tay,Ray,alpha_t,alpha_r,q)
        for i in range(num_iter_end +1, len(nodes_action)):
            n_start =  nodes_action[i]
            Pay[num_iter],Iay[num_iter,0],Iay[num_iter,1] = _forward_pass_nb(V,steps,n_start,Nay,Tay,Ray,alpha_t,alpha_r,q)
            Lay[num_iter] = [n_end, n_start]
            num_iter +=1
        num_iter_end +=1
  
    return Pay,Iay,Lay

@nb.njit
def _backward_pass_nb(V,steps,n_nodes,n_end,Nay,Tay,Ray,alpha_t,alpha_r,q):
    V[:,-1] = 1000
    V[n_end,-1] = 0
    for k in range(steps-2,-1,-1):
        for n0 in range(n_nodes):
            for a in range(8):
                n1,t1,r1=Nay[n0,a],Tay[n0,a],Ray[n0,a]
                if n1==-1:
                    q[a] = 5000
                else:
                    q[a]=V[n1,k+1]+alpha_t*t1+alpha_r*r1
            V[n0,k] = np.min(q)
    return V

@nb.njit 
def _forward_pass_nb(V,steps,n_start,Nay,Tay,Ray,alpha_t,alpha_r,q):
    k0 = np.argmin(V[n_start,:])
    nodes_ay = np.zeros(steps-k0,dtype=np.int64)
    nodes_ay[0] = n_start
    dt,r=0.,0.
    i=0
    for k in range(k0,steps-1):
        n0=nodes_ay[i]
        for a in range(8):
            n1,t1,r1=Nay[n0,a],Tay[n0,a],Ray[n0,a]
            if n1==-1:
                q[a] = 5000
            else:
                q[a]=V[n1,k+1]+alpha_t*t1+alpha_r*r1
        a1=np.argmin(q)
        i+=1
        nodes_ay[i] = Nay[n0,a1]
        dt += Tay[n0,a1]
        r  += Ray[n0,a1]
    return nodes_ay,dt,r

def plot_costmap(data):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label='Cost')
    plt.title('ROS Costmap')
    plt.show()

def get_time_and_risk_dict_new(riskmap, edges, nodes_dct, squares_edge_dict, resolution, vmax,
                               nodes_neighbors, nodes_action, steps, alpha_t, alpha_r, action_idx_name, 
                               scenario_min, scenario_max):
    rt_edges    = get_rt_edges_new(riskmap,edges,nodes_dct,squares_edge_dict,resolution,vmax)
    
    # print(f'rtedges: {np.max(rt_edges)}')
    Nay,Tay,Ray = get_NRT_arrays(nodes_dct,nodes_neighbors,edges,rt_edges)
    num_couples = math.comb(len(nodes_action), 2)
    Lay,Iay=np.zeros((num_couples,2),dtype=np.int64),np.zeros((num_couples,2)) # Lay:couples of nodes_action; Iay: time and risk for each couple
    Pay=nb.typed.Dict.empty(key_type=nb.core.types.int64, value_type=nb.core.types.int64[:])

    n_nodes=len(nodes_dct.keys())
    V=np.zeros((n_nodes,steps))
    q = np.zeros(8)
    Pay,Iay,Lay = get_times_and_risks_new(V,Pay,Lay,Iay,steps,Nay,Tay,Ray,alpha_t,alpha_r,q,n_nodes, nodes_action)
    navigation_risk_mtx = np.zeros((len(nodes_action),len(nodes_action),4),dtype=np.float64)
    for i in range(Iay.shape[0]):
        n0,n1 = Lay[i,:]
        t ,r  = Iay[i,:]
        n0_name = action_idx_name[n0]
        n1_name = action_idx_name[n1]
        time_min = t - t*scenario_min
        time_max = t + t*scenario_max
        # navigation_risk_mtx[n0_name,n1_name,:] = [time_min, t, time_max, r]
        # navigation_risk_mtx[n1_name,n0_name,:] = [time_min, t, time_max, r]
        navigation_risk_mtx[n0_name,n1_name,:] = [round(time_min,2), round(t,2), round(time_max,2), round(r,2)]
        navigation_risk_mtx[n1_name,n0_name,:] = [round(time_min,2), round(t,2), round(time_max,2), round(r,2)]

    return navigation_risk_mtx

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def get_navigation_risk_mtx(merged_costmap, reduced_map, large_graph_params, action_idx_name, 
                             vmax, action_graph_nodes, action_graph_conversion_dict, 
                             risk_params):
    # dynamic data
    resolution = reduced_map.resolution
    cmap = merged_costmap
    riskmap = get_corrected_riskmap(cmap)

    # static data
    nodes_dct         = {int(key): value for key, value in large_graph_params["nodes_ij"].items()}
    edges             = [tuple(element) for element in large_graph_params["edges"]]
    nodes_neighbors   = {int(key): value for key, value in large_graph_params["nodes_neighbors"].items()}
    squares_edge_dict = {int(key): value for key, value in large_graph_params["squares_edge_dict"].items()}
    nodes_action      = list(action_idx_name.keys())
    
    # paremeters
    steps   = risk_params["steps"]
    alpha_t = risk_params["alpha_t"]
    alpha_r = risk_params["alpha_r"]
    scenario_min = risk_params["scenario_min"]
    scenario_max = risk_params["scenario_max"]

    nav_risk_mtx = get_time_and_risk_dict_new(riskmap, edges, nodes_dct, squares_edge_dict, resolution, 
                                              vmax, nodes_neighbors, nodes_action, steps, alpha_t, alpha_r, 
                                              action_idx_name, scenario_min, scenario_max)
    
    for node in action_graph_nodes: 
        node_idx = action_graph_conversion_dict[node]
        x, y = action_graph_nodes[node]["x"], action_graph_nodes[node]["y"] # node location
        i,j = reduced_map.get_costmap_x_y(x,y)
        risk = riskmap[i,j]
        nav_risk_mtx[node_idx, node_idx, :] = [1,1,1, risk]

    return nav_risk_mtx

