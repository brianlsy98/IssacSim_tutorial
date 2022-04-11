from omni.isaac.examples.base_sample import BaseSample
import carb
import omni.ext
import omni.appwindow
import gc
import numpy as np
from omni.isaac.jackal import Jackal
from omni.isaac.husky import Husky
from omni.isaac.environments.floor7_301 import Floor7_301
from omni.isaac.jackal.controllers import DifferentialController as J_dC
from omni.isaac.husky.controllers import DifferentialController as H_dC
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.viewports import set_camera_view

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._jackalcontroller = None
        self._jackalcommand = [0.0, 0.0]
        self._huskycontroller = None
        self._huskycommand = [0.0, 0.0]
        self._vjackal = 70.0
        self._vhusky = 70.0
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
            Jackal(
                prim_path="/World/jackal",
                name="my_jackal",
                position=np.array([-2100, -900.0, 5.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        ##### --- Husky --- #####
        self._husky = world.scene.add(
            Husky(
                prim_path="/World/husky",
                name="my_husky",
                position=np.array([-2100.0, -1000.0, 14.0]),   
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        from omni.isaac.core.objects import VisualSphere
        self.prev_goal = world.scene.add(
            VisualSphere(
                prim_path="/prev_sphere"+str(1),
                name="prev_sphere"+str(1),
                position=np.array([-2100.0, -1000.0, 35.0]),
                radius=5,
                color=np.array([1.0, 1.0, 0]),
            )
        )
        self.prev_goal = world.scene.add(
            VisualSphere(
                prim_path="/prev_sphere"+str(2),
                name="prev_sphere"+str(2),
                position=np.array([-2100.0, -800.0, 35.0]),
                radius=5,
                color=np.array([1.0, 1.0, 0]),
            )
        )
        world.scene.add_ground_plane(static_friction=0.6, dynamic_friction=0.4, restitution=0.8)
        set_camera_view(eye=np.array([-2400, -1200, 200]), target=np.array([-2000, -900, 0]))
        

        return

    async def setup_post_load(self):
        world = self.get_world()
        self._jackal = self._world.scene.get_object("my_jackal")
        self._jackalcontroller = J_dC(name="jackal_control") 
        self._husky = self._world.scene.get_object("my_husky")
        self._huskycontroller = H_dC(name="husky_control")
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("jackal_husky_simplestep", callback_fn=self._on_sim_step)

        self._jackalcommand = [0.0, 0.0]
        self._huskycommand = [0.0, 0.0]
        self._vjackal = 70.0
        self._vhusky = 70.0
        await self._world.play_async()
        return

    def _on_sim_step(self, step):
        self._jackal.apply_wheel_actions(self._jackalcontroller.forward(command=self._jackalcommand))
        self._husky.apply_wheel_actions(self._huskycontroller.forward(command=self._huskycommand)) 
        if self._vjackal != 70.0 : print("jackal v : " + str(self._vjackal))
        if self._vhusky != 70.0 : print("husky v : " + str(self._vhusky))
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

            if event.input == carb.input.KeyboardInput.I:
                self._huskycommand = [self._vhusky, 0.0]
            if event.input == carb.input.KeyboardInput.K:
                self._huskycommand = [-self._vhusky, 0.0]
            if event.input == carb.input.KeyboardInput.J:
                self._huskycommand = [0.0, (np.pi*self._vhusky) / 100]
            if event.input == carb.input.KeyboardInput.L:
                self._huskycommand = [0.0, (-np.pi*self._vhusky) / 100]
            if event.input == carb.input.KeyboardInput.N:
                self._vhusky = self._vhusky + 1.0
            if event.input == carb.input.KeyboardInput.M:
                if self._vhusky > 0 : self._vhusky = self._vhusky - 1.0

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE: ##TODO##
            self._jackalcommand = [0.0, 0.0]
            self._huskycommand = [0.0, 0.0]

        return True

    async def setup_pre_reset(self):
        self._jackalcontroller.reset()
        self._huskycontroller.reset()        
        self._world.remove_physics_callback("jackal_husky_simplestep")
        return

    async def setup_post_reset(self):
        self._world.add_physics_callback("jackal_husky_simplestep", callback_fn=self._on_sim_step)
        await self._world.play_async()
        return

    def world_cleanup(self):
        self._jackalcontroller = None
        self._huskycontroller = None
        self._sub_keyboard = None
        gc.collect()
        return 


 