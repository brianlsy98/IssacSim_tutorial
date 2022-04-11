from pickle import TRUE
import carb
import omni.ext
import omni.appwindow
import gc
import numpy as np
from omni.isaac.jackal import JackalLidar
from omni.isaac.environments.floor7_301 import Floor7_301
from omni.isaac.jackal.controllers import DifferentialController as J_dC
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.range_sensor import _range_sensor


class HelloWorld2(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._jackalcontroller = None
        self._jackalcommand = [0.0, 0.0]
        self._vjackal = 70.0
        ### LIDAR ###
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        self.lidarPath = "/jackal/chassis_link/Lidar"
        #############
        return

    def setup_scene(self):
        world = self.get_world()
        
        ##### --- 301 BUILDING 7th FLOOR --- #####
        self._floor7_301 = world.scene.add(
            Floor7_301(
                prim_path="/World/floor7_301",
                name="my_floor7_301",
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        ##### --- Jackal --- #####
        self._jackal = world.scene.add(
            JackalLidar(
                prim_path="/World/jackal",
                name="my_jackal",
                position=np.array([-2100, -900.0, 5.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        ##### --- Objects --- #####
        from omni.isaac.core.objects import DynamicSphere
        self.d_sphere1 = world.scene.add(
            DynamicSphere(
                prim_path="/dynamic_sphere"+str(1),
                name="dynamic_sphere"+str(1),
                position=np.array([-2100.0, -1000.0, 35.0]),
                radius=15,
                color=np.array([1.0, 1.0, 0]),
                static_friction = 0.6,
                dynamic_friction = 0.4,
            )
        )
        self.d_sphere2 = world.scene.add(
            DynamicSphere(
                prim_path="/dynamic_sphere"+str(2),
                name="dynamic_sphere"+str(2),
                position=np.array([-2100.0, -800.0, 35.0]),
                radius=15,
                color=np.array([0.0, 1.0, 1.0]),
                static_friction = 0.6,
                dynamic_friction = 0.4,
            )
        )
        from omni.isaac.core.objects import VisualSphere
        from pxr import UsdShade
        self.v_sphere1 = world.scene.add(
            VisualSphere(
                prim_path="/visual_sphere"+str(1),
                name="visual_sphere"+str(1),
                position=np.array([-2200.0, -900.0, 25.0]),
                radius=15,
                color=np.array([1.0, 1.0, 0])
            )
        )

        world.scene.add_ground_plane(static_friction=0.6, dynamic_friction=0.4, restitution=0.8)
        set_camera_view(eye=np.array([-2400, -1200, 200]), target=np.array([-2000, -900, 0]))
        
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
        self._vjackal = 70.0
        await self._world.play_async()

        return

    def _on_sim_step(self, step):
        self._jackal.apply_wheel_actions(self._jackalcontroller.forward(command=self._jackalcommand))
        #if self._vjackal != 70.0 : print("jackal v : " + str(self._vjackal))
        ##### -- LIDAR -- #####
        depth = self.lidarInterface.get_linear_depth_data("/World"+self.lidarPath)
        #zenith = self.lidarInterface.get_zenith_data("/World"+self.lidarPath)
        #azimuth = self.lidarInterface.get_azimuth_data("/World"+self.lidarPath)
        print("depth", depth)
        print(len(depth))
        #print("zenith", zenith)
        #print("azimuth", azimuth)
        """for i in range(len(depth)) :
            if depth[i][0] <= 0.75 :
                self.warnsign = 0
                break
            else :
                self.warnsign = 1"""
        #print("self.warnsign", self.warnsign)
        #print(len(depth))
        #######################
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


 
