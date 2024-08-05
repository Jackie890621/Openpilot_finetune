import numpy as np
from gym import spaces

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    """
    in m/s
    """

    def __init__(self, obs_configs):
        self._parent_actor = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'speed': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'speed_xy': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'forward_speed': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'right_up_speed': spaces.Box(low=-10.0, high=30.0, shape=(2,), dtype=np.float32),
            #'accel': spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
            'accel_xy': spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
            'forward_right_up_accel': spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        velocity = self._parent_actor.vehicle.get_velocity()
        acceleration = self._parent_actor.vehicle.get_acceleration()
        transform = self._parent_actor.vehicle.get_transform()
        forward_vec = transform.get_forward_vector()
        right_vec = transform.get_right_vector()
        up_vec = transform.get_up_vector()

        np_vel = np.array([velocity.x, velocity.y, velocity.z])
        np_accel = np.array([acceleration.x, acceleration.y, acceleration.z])
        np_fvec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
        np_rvec = np.array([right_vec.x, right_vec.y, right_vec.z])
        np_uvec = np.array([up_vec.x, up_vec.y, up_vec.z])

        speed = np.linalg.norm(np_vel)
        speed_xy = np.linalg.norm(np_vel[0:2])
        forward_speed = np.dot(np_vel, np_fvec)
        right_speed = np.dot(np_vel, np_rvec)
        up_speed = np.dot(np_vel, np_uvec)

        accel = np.linalg.norm(np_accel)
        accel_xy = np.linalg.norm(np_accel[0:2])
        forward_accel = np.dot(np_accel, np_fvec)
        right_accel = np.dot(np_accel, np_rvec)
        up_accel = np.dot(np_accel, np_uvec)

        obs = {
            'speed': np.array([speed], dtype=np.float32),
            'speed_xy': np.array([speed_xy], dtype=np.float32),
            'forward_speed': np.array([forward_speed], dtype=np.float32),
            'right_up_speed': np.array([right_speed, up_speed], dtype=np.float32),
            #'accel' : np.array([accel], dtype=np.float32),
            'accel_xy' : np.array([accel_xy], dtype=np.float32),
            'forward_right_up_accel' : np.array([forward_accel, right_accel, up_accel], dtype=np.float32)
        }
        return obs

    def clean(self):
        self._parent_actor = None
