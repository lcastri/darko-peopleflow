#!/usr/bin/env python3

import rospy
import numpy as np
from hrisim_prediction_srvs.srv import GetRiskMap, GetRiskMapRequest

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


class TestNode:

    def __init__(self):
        rospy.init_node('tibur_node_service')

    def ask_matrices_and_plot(self, t):
        
        rospy.wait_for_service('/get_risk_map')
        wps = rospy.get_param("/peopleflow/wps")
        
        get_risk_map = rospy.ServiceProxy('/get_risk_map', GetRiskMap)
        req = GetRiskMapRequest()
        resp = get_risk_map(req)

        n_row = resp.n_waypoint
        n_col = resp.n_steps
        prediction_risk_matrix_names = list(resp.waypoint_ids)
        prediction_risk_matrix = np.array(resp.PDs, dtype=np.float64).reshape((n_row, n_col))

        print(prediction_risk_matrix)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        threshold = 0.5
        colors = ['red'] * 4

        for ax_idx, ax in enumerate(axes):
            
            x = []
            y = []
            r = []
            values = []

            for row, name in enumerate(prediction_risk_matrix_names):

                wp = wps[name]
                x.append(wp['x'])
                y.append(wp['y'])
                r.append(wp['r'])
                values.append(prediction_risk_matrix[row, ax_idx])

            x = np.array(x)
            y = np.array(y)
            r = np.array(r)
            values = np.array(values)

            mask = values > threshold
            x_filt = x[mask]
            y_filt = y[mask]
            r_filt = r[mask]

            patches = []
            for i in range(len(x_filt)):
                circle = Circle((x_filt[i], y_filt[i]), r_filt[i])
                patches.append(circle)

            if patches:
                p = PatchCollection(patches, facecolor=colors[ax_idx], alpha=0.6, edgecolor='black')
                ax.add_collection(p)

            max_ray = np.max(r)

            ax.set_xlim(np.min(x) - max_ray, np.max(x) + max_ray)
            ax.set_ylim(np.min(y) - max_ray, np.max(y) + max_ray)
            # ax.set_aspect('equal')
            ax.set_title(f'Plot {ax_idx + 1}')

        plt.tight_layout()
        fig.savefig(f'/home/hrisim/shared/plot/plot_{t}.png')




test_node = TestNode()

t = 0
while True:
    print(t)
    test_node.ask_matrices_and_plot(t)
    rospy.sleep(5)
    t += 5