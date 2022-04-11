from pickle import TRUE
import carb
import omni.ext
import omni.appwindow
import gc
import numpy as np
from copy import deepcopy
from omni.isaac.jackal import JackalLidar
from omni.isaac.jackal.controllers import DifferentialController as J_dC
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.contact_sensor import _contact_sensor


class RLenvSetup(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._jackalcontroller = None
        self._jackalcommand = [0.0, 0.0]
        self._vjackal = 100.0

        # for environment
        self.pre_goal_dist = 0.0
        self.control_freq = 30
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

        ### LIDAR ###
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        self.lidarPath = "/jackal/chassis_link/Lidar"
        #############

        ### Contact Sensor ###
        self.csInterface = _contact_sensor.acquire_contact_sensor_interface()
        self.csPath = "/World/jackal/chassis_link/collisions"
        ######################

        return


    ### BUILD self.candi_goal_pos_list, self.hazard_pos_list
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

    ### UPDATE GOAL ###
    def updateGoalPos(self):
        self.goal_pos = deepcopy(self.candi_goal_pos_list[0])
        self.candi_goal_pos_list = self.candi_goal_pos_list[1:] + self.candi_goal_pos_list[:1]
        self.goal.set_world_pose(np.array([100*self.candi_goal_pos_list[0][0], 100*self.candi_goal_pos_list[0][1], self.cube_size / 2]))


    def setup_scene(self):
        world = self.get_world()
        self.build()

        from omni.isaac.core.objects import DynamicCuboid
        ##### --- SQUARE ENV that Jackal takes place --- #####
        self.square_env.append(world.scene.add(
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
        self.square_env.append(world.scene.add(
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
        self.square_env.append(world.scene.add(
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
        self.square_env.append(world.scene.add(
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
        self._jackal = world.scene.add(
            JackalLidar(
                prim_path="/World/jackal",
                name="my_jackal",
                position=np.array([0.0, 0.0, 30.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        ##### --- HAZARDS --- #####
        for i in range (0, self.num_hazard):
            self.hazards.append(world.scene.add(
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
        self.goal = world.scene.add(
            VisualCuboid(
                prim_path="/World/goal",
                name="goal",
                position=np.array([100*self.candi_goal_pos_list[0][0], 100*self.candi_goal_pos_list[0][1], self.cube_size / 2]),
                size=np.array([self.cube_size, self.cube_size, self.cube_size]),
                color=np.array([1.0, 1.0, 0.0]), # YELLOW
            )
        )

        world.scene.add_ground_plane(static_friction=0.6, dynamic_friction=0.4, restitution=0.8)
        set_camera_view(eye=np.array([500, 500, 400]), target=np.array([0, 0, 0]))
        
        return

    async def setup_post_load(self):
        world = self.get_world()
        self._jackal = self._world.scene.get_object("my_jackal")
        self._jackalcontroller = J_dC(name="jackal_control")
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("jackal_simplestep", callback_fn=self._on_sim_step)

        self._jackalcommand = [0.0, 0.0]
        self._vjackal = 100.0
        await self._world.play_async()

        return

    def _on_sim_step(self, step):
        previous_jackal_position, _ = self._jackal.get_world_pose()
        self._jackal.apply_wheel_actions(self._jackalcontroller.forward(command=self._jackalcommand))
        #if self._vjackal != 70.0 : print("jackal v : " + str(self._vjackal))
        ##### -- LIDAR -- #####
        depth = self.lidarInterface.get_linear_depth_data("/World"+self.lidarPath)
        print("lidar length : " + str(len(depth)))
        #######################
        ### Contact Sensor ###
        props = _contact_sensor.SensorProperties()
        props.radius = 12   # cover the body tip
        props.minThreshold = 0
        props.maxThreshold = 1000000000000
        props.sensorPeriod = 1 / 100.0
        props.position = carb.Float3(0, 0, 0) # Offset sensor 40cm in X direction from rigid body center
        sensor_handle = self.csInterface.add_sensor_on_body(self.csPath, props)
        readings = self.csInterface.get_sensor_readings(sensor_handle)
        print("csSensor : " +str(readings))
        ######################
        current_jackal_position, _ = self._jackal.get_world_pose()
        goal_world_position, _ = self.goal.get_world_pose()
        
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jackal_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jackal_position)
        
        print("d to goal     : " + str(current_dist_to_goal))
        if (current_dist_to_goal < self.cube_size*np.sqrt(2)): self.updateGoalPos()

        return

    def _sub_keyboard_event(self, event, *args, **kwargs):
        #Handle keyboard events
        #w,s,a,d as arrow keys for jackal movement & z,x as velocity change
        #i,k,j,l as arrow keys for husky movement & n,m as velocity change

        #Args: event (int): keyboard event type
        
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            or event.type == carb.input.KeyboardEventType.KEY_REPEAT
        ):
            if event.input == carb.input.KeyboardInput.W:
                self._jackalcommand = [self._vjackal, 0.0]
            if event.input == carb.input.KeyboardInput.S:
                self._jackalcommand = [-self._vjackal, 0.0]
            if event.input == carb.input.KeyboardInput.A:
                self._jackalcommand = [0.0, (np.pi*self._vjackal) / 100]
            if event.input == carb.input.KeyboardInput.D:
                self._jackalcommand = [0.0, (-np.pi*self._vjackal) / 100]
            if event.input == carb.input.KeyboardInput.Z:
                self._vjackal = self._vjackal + 1.0
            if event.input == carb.input.KeyboardInput.X:
                if self._vjackal > 0 : self._vjackal = self._vjackal - 1.0

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE: ##TODO##
            self._jackalcommand = [0.0, 0.0]

        return True

    async def setup_pre_reset(self):
        self._jackalcontroller.reset()    
        self._world.remove_physics_callback("jackal_simplestep")
        return

    async def setup_post_reset(self):
        self._world.add_physics_callback("jackal_simplestep", callback_fn=self._on_sim_step)
        await self._world.play_async()
        return

    def world_cleanup(self):
        self._jackalcontroller = None
        self._sub_keyboard = None
        gc.collect()
        return 


 
