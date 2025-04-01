import pybullet as p
import pybullet_data
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np






class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        
        friction_thing_N = 0
        self.position_control = False
        self.disable_velocity_control = not self.position_control

        self.physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,0])

        # Set up the environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        planeId = p.loadURDF("/home/a2rlab/Documents/bullet3/data/plane.urdf")

        # robotId = p.loadURDF("/home/a2rlab/Documents/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf", [0, 0, 0], useFixedBase=1)
        self.robotId = p.loadURDF("urdfs/indy7.urdf", [0, 0, 0], useFixedBase=1, flags=p.URDF_USE_INERTIA_FROM_FILE)
        num_joints = p.getNumJoints(self.robotId)
        
        if self.disable_velocity_control:
            p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[i for i in range(num_joints)],
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=friction_thing_N*np.ones(num_joints))
        # set joints to 0
        for i in range(1,7):
            p.resetJointState(self.robotId, i, 0, 0)

        p.setRealTimeSimulation(True)
        p.stepSimulation()

        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.ctrl_subscriber = self.create_subscription(JointState, 'joint_controls', self.ctrl_callback, 10)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100Hz
        
        self.joint_state = JointState()
        self.joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.starttime = self.get_clock().now()
        self.jointindices = [i for i in range(1,7)]
        
    def timer_callback(self):
        current_time = self.get_clock().now().to_msg()
        self.joint_state.header.stamp = current_time

        joint_states = p.getJointStates(self.robotId, self.jointindices)
        
        self.joint_state.position = [j[0] for j in joint_states]
        self.joint_state.velocity = [j[1] for j in joint_states]
        self.publisher.publish(self.joint_state)
    
    def ctrl_callback(self, msg):
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                jointIndices=self.jointindices,
                                controlMode=p.TORQUE_CONTROL,
                                forces=1000.0 * np.ones(6))
        if self.position_control:
            p.setJointMotorControlArray(bodyIndex=self.robotId,
                                        jointIndices=self.jointindices,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=msg.position,
                                        forces=[1e5 for _ in range(6)])
        else:
            p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=self.jointindices,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=np.array(msg.effort))
            print(f'applying controls: {msg.effort}')
        

rclpy.init()
joint_state_publisher = JointStatePublisher()
rclpy.spin(joint_state_publisher)
joint_state_publisher.destroy_node()
rclpy.shutdown()

# while 1:

#     if position_control:
#         p.setJointMotorControlArray(bodyIndex=robotId,
#                                     jointIndices=[i for i in range(1,7)],
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPositions=xss[i,:6],
#                                     forces=[1e5 for _ in range(6)]
#                                     )
#     else:
#         p.setJointMotorControlArray(bodyIndex=robotId,
#                                 jointIndices=[i for i in range(1,7)],
#                                 controlMode=p.TORQUE_CONTROL,
#                                 forces=ctrls[i,:])
#     time.sleep(sim_timestep)