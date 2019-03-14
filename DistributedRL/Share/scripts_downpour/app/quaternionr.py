import numpy as np
from msgpackmixin import MsgpackMixin
import math

class Quaternionr(MsgpackMixin):
    w_val = np.float32(0)
    x_val = np.float32(0)
    y_val = np.float32(0)
    z_val = np.float32(0)

    def __init__(self, x_val = np.float32(0), y_val = np.float32(0), z_val = np.float32(0), w_val = np.float32(1)):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.w_val = w_val
        
    @staticmethod
    def toQuaternion(pitch, roll, yaw):
        t0 = math.cos(yaw * 0.5)
        t1 = math.sin(yaw * 0.5)
        t2 = math.cos(roll * 0.5)
        t3 = math.sin(roll * 0.5)
        t4 = math.cos(pitch * 0.5)
        t5 = math.sin(pitch * 0.5)

        q = Quaternionr()
        q.w_val = t0 * t2 * t4 + t1 * t3 * t5 #w
        q.x_val = t0 * t3 * t4 - t1 * t2 * t5 #x
        q.y_val = t0 * t2 * t5 + t1 * t3 * t4 #y
        q.z_val = t1 * t2 * t4 - t0 * t3 * t5 #z
        return q    