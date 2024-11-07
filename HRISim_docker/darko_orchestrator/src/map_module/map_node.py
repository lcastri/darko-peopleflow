#!/usr/bin/env python3

import rospy
from subscribers import OccupancyGridManager
from topic_manager import PublisherManager
from map_functions import merge_costmap, plot_heatmap
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

rospy.init_node('map_node', anonymous=True)

local_costmap = OccupancyGridManager("/move_base/local_costmap/costmap",True)
global_costmap = OccupancyGridManager("/move_base/global_costmap/costmap",True)

merged_costmap_publisher = PublisherManager("/merged_costmap", OccupancyGrid)

rate = rospy.Rate(1)

# n = 0
while not rospy.is_shutdown():

    final_data = merge_costmap(local_costmap, global_costmap)

    # if n >= 30:
    #     plot_heatmap(final_data)
    #     n = -1

    final_grid = OccupancyGrid()
    final_grid.header = Header()
    final_grid.header.frame_id = "map"
    final_grid.header.stamp = rospy.Time.now()
    final_grid.info = global_costmap._occ_grid_metadata

    final_grid.data = final_data.flatten().tolist()
    merged_costmap_publisher._publish_msg(final_grid)

    # n += 1
    rate.sleep()


