# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import gym
from gym import spaces
import numpy as np
import math

## Lidar data as observation
class JackalEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1000,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from omni.isaac.jackal import JackalLidar
        from omni.isaac.core.objects import DynamicCylinder
        from omni.isaac.environments.floor7_301 import Floor7_301
        from omni.isaac.range_sensor import _range_sensor

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=0.01)
        self._my_world.scene.add_ground_plane(static_friction=0.6, dynamic_friction=0.4, restitution=0.8)
        
        self.floor7_301 = self._my_world.scene.add(
            Floor7_301(
                prim_path="/floor7_301",
                name="my_floor7_301",
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.jackal = self._my_world.scene.add(
            JackalLidar(
                prim_path="/jackal",
                name="my_jackal",
                position=np.array([0.0, 0.0, 10.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.goal = self._my_world.scene.add(
            DynamicCylinder(
                prim_path="/new_cylinder",
                name="visual_cylinder",
                position=np.array([90.0, 0.0, 35]),
                radius=10,
                height=45,
                dynamic_friction=10000,
                static_friction=10000,
                mass=10000,
                color=np.array([1.0, 0, 0]),
            )
        )
        
        ### LIDAR ###
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        self.lidarPath = "/jackal/chassis_link/Lidar"
        #############

        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self._set_camera()
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=30, shape=(1080, 1), dtype=np.float32)
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jackal_position, _ = self.jackal.get_world_pose()
        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction

            self.jackal.apply_wheel_velocity_actions(ArticulationAction(joint_velocities=action * 10.0))
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False

        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True

        goal_world_position, _ = self.goal.get_world_pose()
        current_jackal_position, _ = self.jackal.get_world_pose()

        print(goal_world_position)
        #print(current_jackal_position)
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jackal_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jackal_position)
        print(current_dist_to_goal)
        
        if current_dist_to_goal < 50:
            alpha = 2 * math.pi * np.random.rand()
            r = 90 * math.sqrt(np.random.rand()) + 50
            self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 35]))

        reward = previous_dist_to_goal - current_dist_to_goal
        
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 90 * math.sqrt(np.random.rand()) + 50
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 35]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        depth = self.lidarInterface.get_linear_depth_data(self.lidarPath)
        return depth

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _set_camera(self):
        import omni.kit
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        camera_path = "/jackal/chassis_link/Camera"
        if self.headless:
            viewport_handle = omni.kit.viewport.get_viewport_interface()
            viewport_handle.get_viewport_window().set_active_camera(str(camera_path))
            viewport_window = viewport_handle.get_viewport_window()
            self.viewport_window = viewport_window
            viewport_window.set_texture_resolution(128, 128)
        else:
            viewport_handle = omni.kit.viewport.get_viewport_interface().create_instance()
            new_viewport_name = omni.kit.viewport.get_viewport_interface().get_viewport_window_name(viewport_handle)
            viewport_window = omni.kit.viewport.get_viewport_interface().get_viewport_window(viewport_handle)
            viewport_window.set_active_camera(camera_path)
            viewport_window.set_texture_resolution(128, 128)
            viewport_window.set_window_pos(1000, 400)
            viewport_window.set_window_size(420, 420)
            self.viewport_window = viewport_window
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)
        self._my_world.render()
        self.sd_helper.get_groundtruth(["rgb"], self.viewport_window)
        return    