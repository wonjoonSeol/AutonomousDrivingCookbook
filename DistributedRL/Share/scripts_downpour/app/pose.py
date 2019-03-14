from vector3r import Vector3r
from quaternionr import Quaternionr
from msgpackmixin import MsgpackMixin

class Pose(MsgpackMixin):
    position = Vector3r()
    orientation = Quaternionr()

    def __init__(self, position_val, orientation_val):
        self.position = position_val
        self.orientation = orientation_val
