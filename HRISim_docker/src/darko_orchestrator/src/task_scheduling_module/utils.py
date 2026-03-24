from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt


class global_costamap_reduction:

    def __init__(self,costmap_subscriber,reduced_global_map_parameters):
        
        x_min = reduced_global_map_parameters['x_min']
        x_max = reduced_global_map_parameters['x_max']
        y_min = reduced_global_map_parameters['y_min']
        y_max = reduced_global_map_parameters['y_max']

        self.x_min, self.y_min = costmap_subscriber.get_costmap_x_y(x_min, y_min)
        self.x_max, self.y_max = costmap_subscriber.get_costmap_x_y(x_max, y_max)

        self.data = np.copy(costmap_subscriber._grid_data[self.y_min:self.y_max,self.x_min:self.x_max])

        self.origin_x,self.origin_y  = costmap_subscriber.get_world_x_y(self.x_min,self.y_min)

        self.height, self.width = self.data.shape
        self.resolution = costmap_subscriber.resolution
        self.costmap_subscriber = costmap_subscriber


    def update(self):
        self.data = np.copy(self.costmap_subscriber._grid_data[self.y_min:self.y_max,self.x_min:self.x_max])

    
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
        cx, cy = self.get_costmap_x_y(x, y)
        return self.get_cost_from_costmap_x_y(cx, cy)


    def plot_map(self, bottom_left_x, bottom_left_y, neighborhood_size, action_nodes):
        grid_array = np.array(self.data)
        # grid_array = np.array(self.costmap_subscriber._grid_data)
        # Creare un heatmap
        plt.imshow(grid_array, cmap='Greys', interpolation='nearest')

        # Aggiungi il rettangolo
        rect = Rectangle((bottom_left_x, bottom_left_y), 2*neighborhood_size+1, 2*neighborhood_size+1, linewidth=1, edgecolor='green', facecolor='none')
        print(bottom_left_x, bottom_left_y)
        plt.gca().add_patch(rect)
        
        grid_points_x = []
        grid_points_y = []
        for n in action_nodes:
            print(action_nodes[n]["y"], action_nodes[n]["x"])
            col_index, row_index = self.get_costmap_x_y(action_nodes[n]["x"], action_nodes[n]["y"])
            # row_index, col_index = self.costmap_subscriber.get_costmap_x_y(action_nodes[n]["y"], action_nodes[n]["x"])
            print(row_index, col_index)
            grid_points_x.append(col_index)
            grid_points_y.append(row_index)

        # Aggiungi i punti
        plt.scatter(grid_points_x, grid_points_y, color='red', s = 3)

        plt.colorbar()  # Mostra una barra colori per il riferimento dei valori
        plt.show()

    def plot_map_points(self, bottom_left_x, bottom_left_y, neighborhood_size, action_nodes, i_min, i_max, j_min, j_max, x, y):
        grid_array = np.array(self.data)
        # grid_array = np.array(self.costmap_subscriber._grid_data)
        # Creare un heatmap
        plt.imshow(grid_array, cmap='hot', interpolation='nearest')

        # Aggiungi il rettangolo
        rect = Rectangle((bottom_left_x, bottom_left_y), 2*neighborhood_size, 2*neighborhood_size, linewidth=1, edgecolor='green', facecolor='none')
        print(bottom_left_x, bottom_left_y)
        plt.gca().add_patch(rect)
        
        grid_points_x = []
        grid_points_y = []
        for n in action_nodes:
            print(action_nodes[n]["y"], action_nodes[n]["x"])
            col_index, row_index = self.get_costmap_x_y(action_nodes[n]["x"], action_nodes[n]["y"])
            # row_index, col_index = self.costmap_subscriber.get_costmap_x_y(action_nodes[n]["y"], action_nodes[n]["x"])
            print(row_index, col_index)
            grid_points_x.append(col_index)
            grid_points_y.append(row_index)

        # Aggiungi i punti
        plt.scatter(grid_points_x, grid_points_y, color='red', s = 3)
        
        plt.scatter([j_min], [i_min], color='green', s = 3)
        plt.scatter([j_min], [i_max], color='blue', s = 3)
        plt.scatter([j_max], [i_max], color='black', s = 3)
        plt.scatter([j_max], [i_min], color='pink', s = 3)
        plt.scatter([x], [y], color='black', s = 1)
        print(f"x = {x}, y = {y}")
 

        for x, y in zip(grid_points_x, grid_points_y):
            label = f"({x}, {y})"
            plt.text(x, y, label, fontsize=2, ha='right', va='bottom')

        plt.colorbar()  # Mostra una barra colori per il riferimento dei valori
        plt.show()
