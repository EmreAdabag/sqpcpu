import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import sys
import os
current_dir = os.getcwd()
build_dir = os.path.join(current_dir, 'build')
sys.path.append(build_dir)
import pysqpcpu

# Connect to the physics server
physicsClient = p.connect(p.DIRECT)

friction_thing_N = 0
gravity = True
position_control = False
disable_velocity_control = not position_control


# Set up the environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81*gravity)
# planeId = p.loadURDF("/home/a2rlab/Documents/bullet3/data/plane.urdf")

# robotId = p.loadURDF("/home/a2rlab/Documents/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf", [0, 0, 0], useFixedBase=1)
robotId = p.loadURDF("urdfs/indy7.urdf", [0, 0, 0], useFixedBase=1, flags=p.URDF_USE_INERTIA_FROM_FILE)

num_joints = p.getNumJoints(robotId)
print(f"Number of joints: {num_joints}")

# Print joint info
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    print(f"Joint {i}, Name: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")

if disable_velocity_control:
    p.setJointMotorControlArray(bodyIndex=robotId,
                            jointIndices=[i for i in range(num_joints)],
                            controlMode=p.VELOCITY_CONTROL,
                            forces=friction_thing_N*np.ones(num_joints))

sim_timestep = 0.005
solve_timestep = 0.01
p.setTimeStep(sim_timestep) # Default is 1/240 seconds

# stockt = thneed(urdf_filename="urdfs/indy7.urdf", dt=solve_timestep, max_qp_iters=5)
t = pysqpcpu.Thneed(urdf_filename="urdfs/indy7.urdf", N=32, dt=0.01, max_qp_iters=5)
if not gravity:
    t.gravity_off()

xs = 0 * np.hstack([np.ones(6), np.zeros(6)])
for i in range(1,7):
    p.resetJointState(robotId, i, xs[i-1], 0)

goal = t.eepos(0.8 * np.ones(6))
start = t.eepos(xs[:t.nq])

goal_trace = []
tracelen = 330
for i in range(tracelen):
    # goal_trace.append((1/tracelen)*((tracelen - i)*start + (i)*goal)) # linear interpolation
    goal_trace.append(goal)
goal_trace = np.array(goal_trace).reshape(-1)

all_xs = []
all_ctrl = []

running_fails = 0
successes = 0
for i in range(300):
    all_xs.append(np.array(xs))
    # print(np.linalg.norm(stockt.eepos(xs[:6]) - goal_trace[3*(i+stockt.N-1):3*(i+stockt.N)]))
    

    t.setxs(xs)

    if t.sqp(xs, goal_trace[3*i:3*(i+t.N)]):
        print("FAIL")
        running_fails +=1
        if running_fails==t.N-1:
            raise Exception(f"ran out of traj, successes: {successes}")
    else:
        successes +=1
        running_fails = 0
    
    if position_control and 0:
        o = 1
        p.setJointMotorControlArray(bodyIndex=robotId,
                                    jointIndices=[i for i in range(1,7)],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=t.XU[o*(t.nx+t.nu):o*(t.nx+t.nu)+t.nq], # + 0.3 * np.random.rand(6),
                                    # targetVelocities=t.XU[o*(t.nx+t.nu)+t.nq:o*(t.nx+t.nu)+t.nx],
                                    forces=[1e5 for _ in range(6)]
                                    )
    else:
        o = 0
        trqs = t.XU[o*(t.nx+t.nu)+t.nx:o*(t.nx+t.nu)+(t.nx+t.nu)]
        all_ctrl.append(trqs)
        p.setJointMotorControlArray(bodyIndex=robotId,
                                jointIndices=[i for i in range(1,7)],
                                controlMode=p.TORQUE_CONTROL,
                                forces=trqs)
    
    
    # Step the simulation forward
    for i in range(int(solve_timestep / sim_timestep)):
        p.stepSimulation()
        time.sleep(sim_timestep)
    

    joint_states = p.getJointStates(robotId, [i for i in range(1,7)])
    q = [j[0] for j in joint_states]
    v = [j[1] for j in joint_states]
    torques_applied = [j[3] for j in joint_states] # this is zeroed for torque control


    xs[:t.nq] = q
    xs[t.nq:t.nx] = v
    

all_xs = np.array(all_xs)
all_ctrl = np.array(all_ctrl)

np.save('xs.npy', all_xs)
np.save('ctrl.npy', all_ctrl)

# Disconnect from the physics server
p.disconnect()