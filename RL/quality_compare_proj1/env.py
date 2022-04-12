

# TODO is in getState() and __init()__


import gym
from gym import spaces
import numpy as np
import math
from copy import deepcopy
import carb

## Lidar data as observation
class JackalEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        #********* TODO : simulation timestep 설정 ********#
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1000,
        seed=0,
        headless=True,
        #*************************************************#
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
        from omni.isaac.core.objects import DynamicCuboid
        from omni.isaac.range_sensor import _range_sensor
        from omni.isaac.contact_sensor import _contact_sensor
        from omni.isaac.imu_sensor import _imu_sensor
        
        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=0.01)
        self._my_world.scene.add_ground_plane(static_friction=0.6, dynamic_friction=0.4, restitution=0.8)

        self.time_step = 0.002

        # for environment
        self.pre_goal_dist = 0.0
        self.control_freq = 30
        self.num_time_step = int(1.0/(self.time_step*self.control_freq))
        self.limit_distance = 0.5
        self.limit_bound = 0.0
        self.hazard_size = 0.25*np.sqrt(2.0)
        self.goal_dist_threshold = 0.25
        self.h_coeff = 10.0
        self.max_steps = 1000
        self.cur_step = 0
        self.num_hazard = 8
        self.num_goal = 1
        self.num_candi_goal = 5
        self.hazard_group = 2
        self.num_group = 6

        # for candi pos list
        x_space = np.linspace(-2.25, 2.25, 10)
        y_space = np.linspace(-2.25, 2.25, 10)
        self.candi_pos_list = []
        self.candi_pos_indices = []
        cnt = 0
        for x_pos in x_space:
            for y_pos in y_space:
                if abs(x_pos) < 1.0 and abs(y_pos) < 1.0:
                    continue
                self.candi_pos_list.append([x_pos, y_pos])
                self.candi_pos_indices.append(cnt)
                cnt += 1

        ### GRID WORLD SIZE ###
        self.world_size = 1000
        self.square_env = []    # borders as isaac cuboid(white)
        self.cube_size = self.world_size / 20

        ### OBSTACLE ###
        self.hazards = []       # hazards as isaac cuboid
        self.hazard_pos_list = []
        ################

        self.candi_goal_pos_list = []
        self.build()
        
        ##### --- SQUARE ENV that Jackal takes place --- #####
        self.square_env.append(self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/env/env_left",
                name="env_left",
                position=np.array([-self.world_size/2-5.0, 0.0, 50.0]),
                size=np.array([10.0, self.world_size, 100.0]),
                color=np.array([1.0, 1.0, 1.0]), # white
                static_friction = 100000,
                dynamic_friction = 100000,
                mass = 1000,
            )
        ))
        self.square_env.append(self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/env/env_right",
                name="env_right",
                position=np.array([self.world_size/2+5.0, 0.0, 50.0]),
                size=np.array([10.0, self.world_size, 100.0]),
                color=np.array([1.0, 1.0, 1.0]), # RED
                static_friction = 100000,
                dynamic_friction = 100000,
                mass = 1000,
            )
        ))
        self.square_env.append(self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/env/env_up",
                name="env_up",
                position=np.array([0.0, self.world_size/2+5.0, 50.0]),
                size=np.array([self.world_size, 10.0, 100.0]),
                color=np.array([1.0, 1.0, 1.0]), # RED
                static_friction = 100000,
                dynamic_friction = 100000,
                mass = 1000,
            )
        ))
        self.square_env.append(self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/env/env_down",
                name="env_down",
                position=np.array([0.0, -self.world_size/2-5.0, 50.0]),
                size=np.array([self.world_size, 10.0, 100.0]),
                color=np.array([1.0, 1.0, 1.0]), # RED
                static_friction = 100000,
                dynamic_friction = 100000,
                mass = 1000,
            )
        ))

        ##### --- Jackal --- #####
        self.jackal = self._my_world.scene.add(
            JackalLidar(
                prim_path="/World/jackal",
                name="my_jackal",
                position=np.array([0.0, 0.0, 30.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        
        ##### --- HAZARDS --- #####
        for i in range(0, self.num_hazard):
            self.hazards.append(self._my_world.scene.add(
                DynamicCuboid(
                    prim_path="/World/hazards/hazard"+str(i),
                    name="hazard"+str(i),
                    position=np.array([100*self.hazard_pos_list[i][0], 100*self.hazard_pos_list[i][1], self.cube_size / 2]),
                    size=np.array([self.cube_size, self.cube_size, self.cube_size]),
                    color=np.array([1.0, 0.0, 0.0]), # RED
                    static_friction = 0.6,
                    dynamic_friction = 0.4,
                )
            ))
        ##### --- GOAL --- #####
        from omni.isaac.core.objects import VisualCuboid
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/World/goal",
                name="goal",
                position=np.array([100*self.candi_goal_pos_list[0][0], 100*self.candi_goal_pos_list[0][1], self.cube_size / 2]),
                size=np.array([self.cube_size, self.cube_size, self.cube_size]),
                color=np.array([1.0, 1.0, 0.0]), # YELLOW
            )
        )
        ### LIDAR ###
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        self.lidarPath = "/World/jackal/chassis_link/Lidar"
        #############

        ##### --- Contact Sensor --- #####
        self._cs = _contact_sensor.acquire_contact_sensor_interface()
        self.csPath = "/World/jackal/chassis_link"
        cs_props = _contact_sensor.SensorProperties()
        cs_props.radius = -1  # entire body
        cs_props.minThreshold = 0
        cs_props.maxThreshold = 1000000000000
        cs_props.sensorPeriod = 1 / 100.0
        self._cs_sensor_handle = self._cs.add_sensor_on_body(self.csPath, cs_props)
        self.wall_contact = False
        ##################################

        ##### --- IMU Sensor --- #####
        self._is = _imu_sensor.acquire_imu_sensor_interface()
        self.imuPath = "/World/jackal/chassis_link"
        is_props = _imu_sensor.SensorProperties()
        is_props.position = carb.Float3(0, 0, 0)
        is_props.orientation = carb.Float4(0, 0, 0, 1)
        is_props.sensorPeriod = 1 / 500  # 2ms
        self._is_sensor_handle = self._is.add_sensor_on_body(self.imuPath, is_props)
        ##############################        


        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self._set_camera()
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)


        # for state
        self.angle_interval = 2
        self.angle_range = np.arange(-135.0, 135.0 + self.angle_interval, self.angle_interval)
        self.max_scan_value = 3.0
        self.max_goal_dist = 3.0
        self.scan_value = np.zeros(26, dtype=np.float32)
        self.robot_pose = np.zeros(3)
        self.prev_robot_pose = np.zeros(3)
        self.robot_vel = np.zeros(2)
        self.goal_pos = deepcopy(self.candi_goal_pos_list[0])
        # for action
        self.action = np.zeros(2)
        # state & action dimension
        self.action_dim = 2
        self.state_dim = len(self.scan_value) + len(self.robot_pose) + len(self.robot_vel)
        self.state_dim += 1
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.state_dim), np.inf*np.ones(self.state_dim), dtype=np.float32)
        return

    def getState(self):

        ### Contact Sensor ###
        # csSensor : [ time, force, bool ]
        cs_reading = self._cs.get_sensor_readings(self._cs_sensor_handle)
        print(cs_reading)
        ######################
        self.wall_contact = cs_reading[0][-1]

        ##### IMU Sensor #####
        # isSensor : [ time, lin_acc_x, lin_acc_y, lin_acc_z, ang_vel_x, ang_vel_y, ang_vel_z ]
        is_reading = self._is.get_sensor_readings(self._is_sensor_handle)
        print(is_reading)
        ######################

        robot_acc = is_reading[0][1]
        self.robot_pose, _ = self.jackal.get_world_pose()

        #******** TODO : Robot coordinate에서 속도 *******#
        self.robot_vel[0] = np.array(self.robot_pose - self.prev_robot_pose)*self.control_freq
        #***********************************************#
        
        self.robot_vel[1] = is_reading[0][-1]   # angular vel
        self.prev_robot_pose, _ = self.jackal.get_world_pose()
        
        #********** TODO : Quaternion ***********#
        #  -->  local_pos, qt = self._jackal.get_local_pose() 사용하면 quaternion (w, x, y, z) 구해집니다.
        local_pos, qt = self._jackal.get_local_pose()
        robot_mat = local_pos   # 수정필요
        theta = qt*robot_mat    # 수정필요
        self.robot_pose[2] = theta

        rel_goal_pos = self.goal_pos - self.robot_pose[:2]
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        rel_goal_pos = np.matmul(rot_mat, rel_goal_pos)
        goal_dist = np.linalg.norm(rel_goal_pos)
        goal_dir = rel_goal_pos/(goal_dist + 1e-8)
        #****************************************#

        vel = deepcopy(self.robot_vel)
        scan_value = self.getLidar()
        state = {'goal_dir': goal_dir, 'goal_dist': goal_dist, 'vel': vel, 'acc': robot_acc, 'scan': scan_value}
        return state

    def getFlattenState(self, state):
        goal_dir = state['goal_dir']
        goal_dist = [np.clip(state['goal_dist'], 0.0, self.max_goal_dist)]
        vel = state['vel']
        acc = state['acc']
        scan = 1.0 - (np.clip(state['scan'], 0.0, self.max_scan_value)/self.max_scan_value)
        state = np.concatenate([goal_dir, goal_dist, vel, acc/8.0, scan], axis=0)
        return state

    def step(self, action):

        self.cur_step += 1

        for j in range(self.num_time_step):
            from omni.isaac.core.utils.types import ArticulationAction
            self.jackal.apply_wheel_velocity_actions(ArticulationAction(joint_velocities=action))          
            self._my_world.step(render=False)

        state = self.getState()
        info = {"goal_met": False, 'cost': 0.0, 'num_cv': 0}
        info = {}
        # reward
        goal_dist = state['goal_dist']
        reward = self.pre_goal_dist - goal_dist
        self.pre_goal_dist = goal_dist
        print("goal dist : "+str(goal_dist))
        print("reward    : "+str(reward))
        print("")
        if goal_dist < self.goal_dist_threshold:
            print("goal met!")
            reward += 1.0
            info['goal_met'] = True
            self.updateGoalPos()

        # cv
        num_cv = 0
        hazard_dist = np.min(state['scan'])
        if hazard_dist < self.limit_distance:
            num_cv += 1
        info['num_cv'] = num_cv
        info['cost'] = self.getCost(hazard_dist)

        # done
        done = False
        if self.cur_step >= self.max_steps or self.wall_contact:
            done = True
            temp_num_cv = max(self.max_steps - self.cur_step, 0)
            info['num_cv'] += temp_num_cv

        # add raw state
        info['raw_state'] = state

        return self.getFlattenState(state), reward, done, info

    def reset(self):
        self._my_world.reset()
        self.pre_vel = 0.0
        self.pre_ang_vel = 0.0
        self.action = np.zeros(2)
        self.robot_vel = np.zeros(2)
        self.pre_robot_vel = np.zeros(2)
        self.wall_contact = False
        self.build()
        self.updateGoalPos()
        for i in range(self.num_hazard):
            self.hazards[i].set_world_pose(np.array([100*self.hazard_pos_list[i][0], 100*self.hazard_pos_list[i][1], self.cube_size / 2]))
        state = self.getState()
        self.cur_step = 0
        return self.getFlattenState(state)

    def getLidar(self):
        self._my_world.render()
        depth = self.lidarInterface.get_linear_depth_data(self.lidarPath)
        for i in range(len(self.scan_value)):
            self.scan_value[i] = np.mean(depth[5*i:5*i+11])
        return deepcopy(self.scan_value)

    def getGoalDist(self):
        robot_pos, _ = self.jackal.get_world_pose()
        return np.sqrt(np.sum(np.square(self.goal_pos - robot_pos[:2])))

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

        camera_path = "/World/jackal/chassis_link/Camera"
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

    def build(self):
        while True:
            sampled_candi_indices = np.random.choice(self.candi_pos_indices, self.num_hazard + self.num_candi_goal, replace=False)
            hazard_pos_list = [np.array(self.candi_pos_list[idx]) for idx in sampled_candi_indices[:self.num_hazard]]
            candi_goal_pos_list = [np.array(self.candi_pos_list[idx]) for idx in sampled_candi_indices[self.num_hazard:]]
            good_goal_pos_list = []
            for candi_goal_pos in candi_goal_pos_list:
                is_good = True
                for hazard_pos in hazard_pos_list:
                    hazard_dist = np.linalg.norm(hazard_pos - candi_goal_pos) - self.hazard_size
                    if hazard_dist <= self.limit_distance + 2.0/self.h_coeff:
                        is_good = False
                        break
                if is_good:
                    good_goal_pos_list.append(candi_goal_pos)

            if len(good_goal_pos_list) >= 3:
                min_dist = np.inf
                for goal_idx in range(len(good_goal_pos_list) - 1):
                    dist = np.linalg.norm(good_goal_pos_list[goal_idx] - good_goal_pos_list[goal_idx+1])
                    if dist < min_dist:
                        min_dist = dist
                if min_dist > 2.0:
                    # 2D
                    self.candi_goal_pos_list = deepcopy(good_goal_pos_list)
                    self.hazard_pos_list = deepcopy(hazard_pos_list)
                    break
                else:
                    pass
            else:
                pass

    def updateGoalPos(self):
        self.goal_pos = deepcopy(self.candi_goal_pos_list[0])
        self.candi_goal_pos_list = self.candi_goal_pos_list[1:] + self.candi_goal_pos_list[:1]
        self.goal.set_world_pose(np.array([100*self.candi_goal_pos_list[0][0], 100*self.candi_goal_pos_list[0][1], self.cube_size / 2]))
        self.pre_goal_dist = self.getGoalDist()
