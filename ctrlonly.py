import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys 
import os
current_dir = os.getcwd()
build_dir = os.path.join(current_dir, 'build')
sys.path.append(build_dir)
import pysqpcpu



class TorqueCalculator(Node):
    def __init__(self):
        super().__init__('torque_calculator')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            10)
        self.publisher = self.create_publisher(
            JointState, 
            'joint_commands', 
            10)
        self.ctrl_msg = JointState()
        self.ctrl_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        self.t = pysqpcpu.Thneed(urdf_filename="urdfs/indy7.urdf", N=32, dt=0.01, max_qp_iters=5)

        self.joint_positions = None
        self.joint_velocities = None
        self.goal_trace = np.tile(self.t.eepos(0.8 * np.ones(6)), self.t.N)
        self.successes = 0
        self.running_fails = 0

    def joint_callback(self, msg):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)
        self.xs = np.hstack([self.joint_positions, self.joint_velocities])

        self.t.setxs(self.xs)

        s = time.time()
        if self.t.sqp(self.xs, self.goal_trace):
            print("FAIL")
            self.running_fails +=1
            if self.running_fails==self.t.N-1:
                raise Exception(f"ran out of traj, successes: {self.successes}")
        else:
            self.successes +=1
            self.running_fails = 0
        print(f"sqp time: {1000 * (time.time() - s)} ms")

        # Publish torques
        self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.ctrl_msg.position = list(self.t.XU[1*(self.t.nx+self.t.nu):1*(self.t.nx+self.t.nu)+self.t.nq])
        self.ctrl_msg.velocity = [0.0] * self.t.nq
        self.ctrl_msg.effort = list(self.t.XU[self.t.nx:(self.t.nx+self.t.nu)])
        self.publisher.publish(self.ctrl_msg)
        # self.get_logger().info(f"Publishing torques: {[round(t, 2) for t in torques]}")
        

def main(args=None):
    rclpy.init(args=args)
    torque_calculator = TorqueCalculator()
    rclpy.spin(torque_calculator)
    torque_calculator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
