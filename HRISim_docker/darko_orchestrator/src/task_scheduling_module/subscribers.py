import rospy
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int64MultiArray, Float64MultiArray, Int64MultiArray
import numpy as np
from darko_orchestrator.msg import ScenarioList, Scenario, State

class OccupancyGridManager(object):
    
    def __init__(self, topic, subscribe_to_updates=False):
        # OccupancyGrid starts on lower left corner
        self._grid_data = None
        self._occ_grid_metadata = None
        self._reference_frame = None
        self._sub = rospy.Subscriber(topic, OccupancyGrid,
                                     self._occ_grid_cb,
                                     queue_size=1)
        if subscribe_to_updates:
            self._updates_sub = rospy.Subscriber(topic + '_updates',
                                                 OccupancyGridUpdate,
                                                 self._occ_grid_update_cb,
                                                 queue_size=1)
            
        rospy.loginfo("Waiting for '" + str(self._sub.resolved_name) + "'...")

        while self._occ_grid_metadata is None and self._grid_data is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

    @property
    def resolution(self):
        return self._occ_grid_metadata.resolution

    @property
    def width(self):
        return self._occ_grid_metadata.width

    @property
    def height(self):
        return self._occ_grid_metadata.height

    @property
    def origin(self):
        return self._occ_grid_metadata.origin

    @property
    def reference_frame(self):
        return self._reference_frame

    def _occ_grid_cb(self, data):
        self._occ_grid_metadata = data.info
        self._grid_data = np.array(data.data,
                                   dtype=np.int64).reshape(data.info.height,
                                                          data.info.width)
        self._reference_frame = data.header.frame_id

    def _occ_grid_update_cb(self, data):
         
        if self._grid_data is not None:

            data_np = np.array(
                data.data,
                dtype=np.int64
            ).reshape(
                data.height,
                data.width
            )

            self._grid_data[data.y:data.y + data.height, data.x:data.x + data.width] = data_np

    def get_world_x_y(self, costmap_x, costmap_y):
        world_x = costmap_x * self.resolution + self.origin.position.x
        world_y = costmap_y * self.resolution + self.origin.position.y
        return world_x, world_y

    def get_costmap_x_y(self, world_x, world_y):
        costmap_x = int(
            round((world_x - self.origin.position.x) / self.resolution))
        costmap_y = int(
            round((world_y - self.origin.position.y) / self.resolution))
        return costmap_x, costmap_y

    def get_cost_from_world_x_y(self, x, y):
        cx, cy = self.get_costmap_x_y(x, y)
        try:
            return self.get_cost_from_costmap_x_y(cx, cy)
        except IndexError as e:
            raise IndexError("Coordinates out of grid (in frame: {}) x: {}, y: {} must be in between: [{}, {}], [{}, {}]. Internal error: {}".format(
                self.reference_frame, x, y,
                self.origin.position.x,
                self.origin.position.x + self.height * self.resolution,
                self.origin.position.y,
                self.origin.position.y + self.width * self.resolution,
                e))

    def get_cost_from_costmap_x_y(self, x, y):
        if self.is_in_gridmap(x, y):
            return self._grid_data[y][x]
        else:
            raise IndexError(
                "Coordinates out of gridmap, x: {}, y: {} must be in between: [0, {}], [0, {}]".format(
                    x, y, self.height, self.width))

    def is_in_gridmap(self, x, y):
        if -1 < x < self.width and -1 < y < self.height:
            return True
        else:
            return False

class AmclPoseManager(object):
    def __init__(self):
        self._topic = "/darko/amcl_pose"
        self._pose_data = None
        self._reference_frame = None
        self._sub = rospy.Subscriber(self._topic, PoseWithCovarianceStamped,self._amcl_pose_cb,queue_size=1)
        while self._pose_data is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

    @property
    def position(self):
        return self._pose_data.pose.position

    @property
    def orientation(self):
        return self._pose_data.pose.orientation

    @property
    def covariance_matrix(self):
        return self._pose_data.covariance

    def _amcl_pose_cb(self, data):
        self._pose_data = data.pose
        self._reference_frame = data.header.frame_id

    def get_position_xy(self):
        return np.asarray([self.position.x,self.position.y])

class RiskMtxSubscriber(object):
    def __init__(self,topic):
        self._topic = topic
        self._risk_data = None
        self._mtx_shape  = None
        self._sub = rospy.Subscriber(self._topic, Int64MultiArray,self._risk_mtx_cb,queue_size=1)
        
    def _risk_mtx_cb(self, data):
        if self._mtx_shape is None:
            layout = data.layout
            ndims = len(layout.dim)
            self._mtx_shape =  [layout.dim[i].size for i in range(ndims)]
        self._risk_data = np.array(data.data,dtype=np.int64).reshape(self._mtx_shape)

class RiskMtxSubscriberFloat(object):
    def __init__(self,topic):
        self._topic = topic
        self._risk_data = None
        self._mtx_shape  = None
        self._sub = rospy.Subscriber(self._topic, Float64MultiArray,self._risk_mtx_cb,queue_size=1)
        
    def _risk_mtx_cb(self, data):
        if self._mtx_shape is None:
            layout = data.layout
            ndims = len(layout.dim)
            self._mtx_shape =  [layout.dim[i].size for i in range(ndims)]
        self._risk_data = np.array(data.data,dtype=np.float64).reshape(self._mtx_shape)

class ScenariosSubscriber(object):
    def __init__(self,topic_scenarios, topic_probabilities):
        self._topic_scenarios = topic_scenarios
        self._topic_probabilities = topic_probabilities
        self._scenarios_data = None
        self._probabilities_data = None
        self._sub_scenarios = rospy.Subscriber(self._topic_scenarios, ScenarioList,self._scenarios_cb,queue_size=1)
        self._sub_probabilities = rospy.Subscriber(self._topic_probabilities, Float64MultiArray,self._probabilities_cb,queue_size=1)
        
    def _scenarios_cb(self, data):
        self._scenarios_data = data.data

    def _probabilities_cb(self, data):
        self._probabilities_data = data.data
