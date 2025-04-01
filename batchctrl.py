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
        
        self.batch_size = 1
        self.num_threads = 1
        self.dt = 0.01

        self.t = pysqpcpu.BatchThneed(urdf_filename="urdfs/indy7.urdf", batch_size=self.batch_size, N=32, dt=self.dt, max_qp_iters=5, num_threads=self.num_threads)

        self.last_state_msg = None
        # self.joint_positions = None
        # self.joint_velocities = None
        self.last_results = None
        

        self.goal_trace = np.tile(self.t.eepos(0.8 * np.ones(6)), self.t.N)
        # self.successes = 0
        # self.running_fails = 0
        # self.last_joint_callback = None

        self.xs_batch = [None] * self.batch_size
        self.eepos_g_batch = [self.goal_trace] * self.batch_size
        self.fext_batch = np.zeros((self.batch_size, 3))

    def joint_callback(self, msg):

        self.xs = np.hstack([np.array(msg.position), np.array(msg.velocity)])
        
        if self.last_state_msg is not None:
            '''
                using 2 state messages, find best result
            '''
            # get offset into results array
            m1_time = self.last_state_msg.header.stamp.sec + self.last_state_msg.header.stamp.nanosec * 1e-9
            m2_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            time_since_planning_start = m2_time - m1_time
            full_step_offset = int(time_since_planning_start / self.dt)
            inter_step_offset = time_since_planning_start - full_step_offset * self.dt
            print(f"full_step_offset: {full_step_offset}, inter_step_offset: {inter_step_offset} s")
            
            best_result = None
            best_error = np.inf
            for i, result in enumerate(self.last_results):
                # get expected state for each result
                s1 = result[full_step_offset * (self.t.nx+self.t.nu):(full_step_offset+1)*(self.t.nx+self.t.nu) - self.t.nu]
                s2 = result[(full_step_offset+1) * (self.t.nx+self.t.nu):(full_step_offset+2)*(self.t.nx+self.t.nu) - self.t.nu]
                expected_state = s1 + inter_step_offset * (s2 - s1) / self.dt
                error = np.linalg.norm(expected_state - self.xs)
                if error < best_error:
                    best_error = error
                    best_result = result
            
            # Publish torques from best matching result (offset by full_step_offset)
            self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
            self.ctrl_msg.position = list(best_result[full_step_offset*(self.t.nx+self.t.nu):full_step_offset*(self.t.nx+self.t.nu) + self.t.nq])
            # self.ctrl_m 1sg.velocity = self.joint_velocities
            self.ctrl_msg.effort = list(best_result[full_step_offset*(self.t.nx+self.t.nu) + self.t.nx:(full_step_offset+1)*(self.t.nx+self.t.nu)])
            self.publisher.publish(self.ctrl_msg)

        # self.joint_positions = 
        # self.joint_velocities = 

        for i in range(self.batch_size):
            self.xs_batch[i] = self.xs
            self.eepos_g_batch[i] = self.goal_trace
            self.fext_batch[i] = np.zeros(3)
            # self.fext_batch[i] = np.random.normal(0.0, 2.0, 3) # random forces on end effector

        self.t.batch_update_xs(self.xs_batch)
        self.t.batch_set_fext(self.fext_batch)
        s = time.time()
        self.t.batch_sqp(self.xs_batch, self.eepos_g_batch) # run batch sqp
        print(f"batch sqp time: {1000 * (time.time() - s)} ms")

        self.last_results = self.t.get_results()
        self.last_state_msg = msg
        # for i, result in enumerate(results):
        #     print(f"Result {i}: {result[:13]} ...")

        # self.last_joint_callback = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # self.get_logger().info(f"Publishing torques: {[round(t, 2) for t in torques]}")

def main(args=None):
    try:
        rclpy.init(args=args)
        torque_calculator = TorqueCalculator()
        while rclpy.ok():
            rclpy.spin_once(torque_calculator, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        torque_calculator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
