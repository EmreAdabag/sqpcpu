import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
cartpole = p.loadURDF("cartpole.urdf")
p.setJointMotorControl2(cartpole,1,p.VELOCITY_CONTROL,force=0)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

while (1):
    p.setJointMotorControl2(cartpole,1,p.TORQUE_CONTROL,force=10)
    js = p.getJointState(cartpole, 1)
    print("position=", js[0], "velocity=", js[1])
    time.sleep(0.01)
