from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
import numpy as np


class DifferentialController(BaseController):
    """Controller uses unicycle model for a diffrential drive

        Args:
            name (str): [description]
            wheel_radius (float): Radius of left and right wheels in cms
            wheel_base (float): Distance between left and right wheels in cms
        """
    def __init__(self, name: str, wheel_radius=8, wheel_base=18) -> None:
        super().__init__(name)
        self._wheel_radius = wheel_radius
        self._wheel_base = wheel_base
        return

    def forward(self, command: np.ndarray) -> ArticulationAction:
        if isinstance(command, list):
            command = np.array(command)
        if command.shape[0] != 2:
            raise Exception("command should be of length 2")
        joint_velocities = [0.0, 0.0, 0.0, 0.0]
        joint_velocities[0] = ((2 * command[0]) - (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        joint_velocities[1] = ((2 * command[0]) + (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        joint_velocities[2] = (2 * command[0] - (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        joint_velocities[3] = (2 * command[0] + (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        return ArticulationAction(joint_velocities=joint_velocities)

    def reset(self) -> None:
        """[summary]
        """
        return



