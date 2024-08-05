import numpy as np
import copy
import weakref
import carla
from queue import Queue, Empty
from gym import spaces
from collections import deque
import math

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from carla_gym.core.obs_manager.camera.lane_markings import LaneMarkings

#number_of_lanepoints = 33
#meters_per_frame = 3.0
X_IDXS = [0.    ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
          6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
          27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
          60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
          108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
          168.75  , 180.1875, 192.]

class ObsManager(ObsManagerBase):
    """
    Template configs:
    obs_configs = {
        "module": "camera.rgb",
        "location": [-5.5, 0, 2.8],
        "rotation": [0, -15, 0],
        "frame_stack": 1,
        "width": 1920,
        "height": 1080
    }
    frame_stack: [Image(t-2), Image(t-1), Image(t)]
    """

    def __init__(self, obs_configs):

        self._sensor_type = 'camera.rgb'

        self._height = obs_configs['height']
        self._width = obs_configs['width']
        self._fov = obs_configs['fov']
        self._channels = 4

        location = carla.Location(
            x=float(obs_configs['location'][0]),
            y=float(obs_configs['location'][1]),
            z=float(obs_configs['location'][2]))
        rotation = carla.Rotation(
            roll=float(obs_configs['rotation'][0]),
            pitch=float(obs_configs['rotation'][1]),
            yaw=float(obs_configs['rotation'][2]))

        self._camera_transform = carla.Transform(location, rotation)

        self._sensor = None
        self._queue_timeout = 10.0
        self._image_queue = None
        self.seg_queue = None
        self.world_map = None
        self.seg_cam = None
        self.detection_sensor = None
        self.lead_actor = None
        self.lead_dis = -1
        #self.obstacle_queue = None

        super(ObsManager, self).__init__()

    def _define_obs_space(self):

        self.obs_space = spaces.Dict({
            'frame': spaces.Discrete(2**32-1),
            'data': spaces.Box(
                low=0, high=255, shape=(self._height, self._width, self._channels), dtype=np.uint8),
            'location' : spaces.Box(
                low=-5000, high=5000, shape=(3,), dtype=np.float32),
            'rotation' : spaces.Box(
                low=-360, high=360, shape=(3,), dtype=np.float32),
            'lanelines' : spaces.Box(
                low=-5000, high=5000, shape=(4,33,3), dtype=np.float32),
            #'roadedges' : spaces.Box(
            #    low=-5000, high=5000, shape=(2,33,3), dtype=np.float32),
            'available_lanelines' : spaces.Box(
                low=0, high=1, shape=(4,33), dtype=np.uint8),
            #'available_roadedges' : spaces.Box(
            #    low=0, high=1, shape=(4,33), dtype=np.uint8),
            'lead_distance' : spaces.Box(
                low=0, high=250, shape=(1,), dtype=np.float32),
            'lead_speed' : spaces.Box(
                low=-10, high=30, shape=(1,), dtype=np.float32),
            'lead_accel' : spaces.Box(
                low=-10, high=10, shape=(1,), dtype=np.float32),
        })

    def attach_ego_vehicle(self, parent_actor):
        init_obs = np.zeros([self._height, self._width, self._channels], dtype=np.uint8)
        self._image_queue = Queue()
        self.seg_queue = Queue()
        #self.obstacle_queue = Queue()

        self._world = parent_actor.vehicle.get_world()

        bp = self._world.get_blueprint_library().find("sensor."+self._sensor_type)
        bp.set_attribute('image_size_x', str(self._width))
        bp.set_attribute('image_size_y', str(self._height))
        bp.set_attribute('fov', str(self._fov))
        # set in leaderboard
        bp.set_attribute('lens_circle_multiplier', str(3.0))
        bp.set_attribute('lens_circle_falloff', str(3.0))
        bp.set_attribute('chromatic_aberration_intensity', str(0.5))
        bp.set_attribute('chromatic_aberration_offset', str(0))

        self._sensor = self._world.spawn_actor(bp, self._camera_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self.world_map = self._world.get_map()
        self._sensor.listen(lambda image: self._parse_image(weak_self, image))

        seg_bp = self._world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(self._width))
        seg_bp.set_attribute('image_size_y', str(self._height))
        seg_bp.set_attribute('fov', str(self._fov))
        seg_bp.set_attribute('lens_circle_multiplier', str(3.0))
        seg_bp.set_attribute('lens_circle_falloff', str(3.0))

        self.seg_cam = self._world.spawn_actor(seg_bp, self._camera_transform, attach_to=parent_actor.vehicle)
        self.seg_cam.listen(lambda seg_img: self._seg_callback(weak_self, seg_img))

        detection_bp = self._world.get_blueprint_library().find('sensor.other.obstacle')
        detection_bp.set_attribute('distance','200.0')
        detection_bp.set_attribute('hit_radius','0.5')
        detection_bp.set_attribute('only_dynamics','False')
        #detection_bp.set_attribute('sensor_tick','0.05')
        #detection_bp.set_attribute('debug_linetrace','True')
        transform = carla.Transform(parent_actor.vehicle.bounding_box.location+carla.Location(x=(parent_actor.vehicle.bounding_box.extent.x)))
        self.detection_sensor = self._world.spawn_actor(detection_bp, transform, attach_to=parent_actor.vehicle)
        self.detection_sensor.listen(lambda obstacle: self.obstacle_callback(weak_self, obstacle))



    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        assert self._image_queue.qsize() <= 1
        assert self.seg_queue.qsize() <= 1
        #assert self.obstacle_queue.qsize() <= 1

        try: 
            frame, data, location, rotation, lanelines3d22d, roadedges3d22d, lanelines3d, roadedges3d= self._image_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('RGB sensor took too long!')
        try: 
            frame, seg_data = self.seg_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('seg sensor took too long!')

        if self.lead_actor:
            lead_distance = self.lead_dis
            vel = self.lead_actor.get_velocity()
            lead_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            accel = self.lead_actor.get_acceleration()
            lead_accel = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        else :
            lead_distance = -1
            lead_speed = -1
            lead_accel = -1

        lanelines = np.float32(lanelines3d)
        roadedges = np.float32(roadedges3d)
        available_lanelines = []
        available_roadedges = []
        lanemarkings = LaneMarkings()
        for lanelist in lanelines3d22d:
            lanelist = lanemarkings.calculate2DLanepoints(self._sensor, lanelist)
            lanelist = lanemarkings.filter2DLanepoints(lanelist, seg_data)
            available_lanelines.append(lanelist)
        for edgelist in roadedges3d22d:
            edgelist = lanemarkings.calculate2DLanepoints(self._sensor, edgelist)
            edgelist = lanemarkings.filter2DLanepoints(edgelist, seg_data)
            available_roadedges.append(edgelist)



        obs = {'frame': np.array([frame]),
               'data': data,
               'location' : location,
               'rotation' : rotation,
               'lanelines' : lanelines,
               #'roadedges' : roadedges,
               'available_lanelines' : available_lanelines,
               #'available_roadedges' : available_roadedges,
               'lead_distance' : np.array([lead_distance]),
               'lead_speed' : np.array([lead_speed]),
               'lead_accel' : np.array([lead_accel])}

        return obs

    def clean(self):
        if self._sensor and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
        if self.seg_cam and self.seg_cam.is_alive:
            self.seg_cam.stop()
            self.seg_cam.destroy()
        if self.detection_sensor and self.detection_sensor.is_alive:
            self.detection_sensor.stop()
            self.detection_sensor.destroy()
        self._sensor = None
        self._world = None
        self.world_map = None
        self.seg_cam = None
        self.detection_sensor = None

        self._image_queue = None
        self.seg_queue = None
        self.lead_actor = None
        self.lead_dis = -1
        #self.obstacle_queue = None

    @staticmethod
    def _parse_image(weak_self, carla_image):
        self = weak_self()

        np_img = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))

        np_img = copy.deepcopy(np_img)

        np_img = np.reshape(np_img, (carla_image.height, carla_image.width, 4))
        np_img = np_img[:, :, :3]
        np_img = np_img[:, :, ::-1]

        # np_img = np.moveaxis(np_img, -1, 0)
        # image = cv2.resize(image, (self._res_x, self._res_y), interpolation=cv2.INTER_AREA)
        # image = np.float32
        # image = (image.astype(np.float32) - 128) / 128
        lanelines3d = []
        roadedges3d = []
        lanelines3d22d = []
        roadedges3d22d = []
        lane_markings = []
        road_edge_markings = []

        way_point = self.world_map.get_waypoint(carla_image.transform.location)
        waypoint_list = deque(maxlen=len(X_IDXS))
        waypoint_list.append(way_point)
        for i in range(len(X_IDXS)-1):
            waypoint_list.append(way_point.next(X_IDXS[i+1])[0])
        lanemarkings = LaneMarkings()
        for lanepoint in waypoint_list:
            lane_markings = lanemarkings.calculate3DLanepoints(lanepoint)
            road_edge_markings = lanemarkings.calculate3DRoadEdgePoints(lanepoint)

        for i in range(4):
            a2d_list=[]
            a3d_list=[]
            for j in range(33):
                lane_vector = lane_markings[i].popleft()
                if lane_vector :
                    a2d_list.append([lane_vector.x, lane_vector.y, lane_vector.z, 1.0])
                    a3d_list.append([lane_vector.x, lane_vector.y, lane_vector.z])
                else :
                    a2d_list.append([None,None,None,None])
                    a3d_list.append([0.0, 0.0 ,0.0])
            lanelines3d22d.append(a2d_list)
            lanelines3d.append(a3d_list)

        for i in range(2):
            a2d_list=[]
            a3d_list=[]
            for j in range(33):
                lane_vector = road_edge_markings[i].popleft()
                if lane_vector :
                    a2d_list.append([lane_vector.x, lane_vector.y, lane_vector.z, 1.0])
                    a3d_list.append([lane_vector.x, lane_vector.y, lane_vector.z])
                else :
                    a2d_list.append([None,None,None,None])
                    a3d_list.append([0.0, 0.0 ,0.0])
            roadedges3d22d.append(a2d_list)
            roadedges3d.append(a3d_list)

        self._image_queue.put((carla_image.frame, np_img, [carla_image.transform.location.x, carla_image.transform.location.y, carla_image.transform.location.z],
         [carla_image.transform.rotation.roll, carla_image.transform.rotation.pitch, carla_image.transform.rotation.yaw], 
         lanelines3d22d, roadedges3d22d,lanelines3d,roadedges3d))


    @staticmethod
    def _seg_callback(weak_self, seg_image):
        self = weak_self()
        np_img = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))

        np_img = copy.deepcopy(np_img)

        np_img = np.reshape(np_img, (seg_image.height, seg_image.width, 4))
        np_img = np_img[:, :, :3]
        np_img = np_img[:, :, ::-1]

        self.seg_queue.put((seg_image.frame, np_img))


    @staticmethod
    def obstacle_callback(weak_self, obstacle):
        self = weak_self()
        if obstacle.other_actor.type_id.startswith('vehicle.'):
            self.lead_actor = obstacle.other_actor
            self.lead_dis = obstacle.distance
        else :
            self.lead_actor = None
            self.lead_dis = -1
        '''
        if obstacle.other_actor.type_id.startswith('vehicle.'):
            vel = obstacle.other_actor.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            accel = obstacle.other_actor.get_acceleration()
            accel = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
            self.obstacle_queue.put((obstacle.frame, obstacle.distance, speed, accel))
        else :
            self.obstacle_queue.put((obstacle.frame, -1, -1, -1))
        '''