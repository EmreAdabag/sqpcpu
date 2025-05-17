from pinocchio_template import thneed
import time
import math
import numpy as np
import sys
import os


friction_thing_N = 0
gravity = True
position_control = False
disable_velocity_control = not position_control


t = thneed(xml_filename="urdfs/frankapanda/mjx_panda.xml", eepos_frame_name="hand", N=32, dt=0.01, max_qp_iters=5)
print(t.eepos_frame_id)

xs = 0.0 * np.hstack([np.ones(t.nq), np.zeros(t.nv)])
goal = 0.8 * np.ones(3)
goal_trace = np.tile(goal, t.N)



t.setxs(xs)
print("sqp success: ", not t.sqp(xs, goal_trace))
print("XU: ", t.XU)

