import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
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
        self.goal_sub = self.create_subscription(
            PoseStamped,
            'goal',
            self.goal_callback,
            10
        )
        self.publisher = self.create_publisher(
            JointState, 
            'joint_commands', 
            10)
        self.ctrl_msg = JointState()
        self.ctrl_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        self.batch_size = 6
        self.num_threads = 6
        self.dt = 0.01
        self.fext_timesteps = 32

        self.t = pysqpcpu.BatchThneed(urdf_filename="urdfs/indy7.urdf", batch_size=self.batch_size, N=32, dt=self.dt, max_qp_iters=5, num_threads=self.num_threads, fext_timesteps=self.fext_timesteps)

        self.last_state_msg = None
        # self.joint_positions = None
        # self.joint_velocities = None
        self.last_results = None
        self.last_best_result_idx = 0
        

        self.goal_trace = np.tile(self.t.eepos(0.0 * np.ones(6)), self.t.N)
        # self.successes = 0
        # self.running_fails = 0
        # self.last_joint_callback = None

        self.xs_batch = [None] * self.batch_size
        self.eepos_g_batch = [self.goal_trace] * self.batch_size
        # self.fext_batch = 50.0 * np.ones((self.batch_size, 3))
        self.fext_batch = np.random.normal(0.0, 2.0, (self.batch_size, 3))
        self.fext_batch[0] = np.array([0.0, 0.0, 0.0])
        print(self.fext_batch)
        self.t.batch_set_fext(self.fext_batch)
        self.last_xs = None
        self.last_u = None
        self.lastnoiseinput = np.zeros(self.t.nu)
    
    def NOISE(self):
        # self.lastnoiseinput += np.random.normal(0.0, 0.1, self.t.nu)
        return self.lastnoiseinput
    
    def goal_callback(self, msg):
        print('received pose: ', np.array(msg.pose.position))
        self.goal_trace = np.tile(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]), self.t.N)
        self.eepos_g_batch = [self.goal_trace] * self.batch_size

    def joint_callback(self, msg):

        self.xs = np.hstack([np.array(msg.position), np.array(msg.velocity)])
        
            

        # self.joint_positions = 
        # self.joint_velocities = 

        for i in range(self.batch_size):
            self.xs_batch[i] = self.xs
        self.t.batch_update_xs(self.xs_batch)

        s = time.time()
        self.t.batch_sqp(self.xs_batch, self.eepos_g_batch) # run batch sqp
        print(f"batch sqp time: {1000 * (time.time() - s)} ms")

        
        
        '''
            using the last applied control, identify the best predictor of the next state
            apply last applied control to last message state (with fext) and get expected next state
            closest result is the best predictor for next result
        '''
        if self.last_state_msg is not None:
            m1_time = self.last_state_msg.header.stamp.sec + self.last_state_msg.header.stamp.nanosec * 1e-9
            m2_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            step_duration = m2_time - m1_time
            # get prediction based on last applied control
            predictions = self.t.predict_fwd(self.last_xs, self.last_u, step_duration)

            best_tracker_idx = None
            best_error = np.inf
            for i, result in enumerate(predictions):
                # get expected state for each result
                error = np.linalg.norm(result - self.xs)
                if error < best_error:
                    best_error = error
                    best_tracker_idx = i
            
            # resample fexts around the best result
            # self.fext_batch[:] = self.fext_batch[best_tracker_idx]
            # self.fext_batch = np.random.normal(self.fext_batch, 0.05)
            # self.t.batch_set_fext(self.fext_batch)

        else:
            best_tracker_idx = 0

        print(f'most accurate force: {self.fext_batch[best_tracker_idx]}')
        
        best_result = self.t.get_results()[best_tracker_idx]
        # Publish torques from batch result that best matched dynamics on the last step
        self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.ctrl_msg.position = [0.0] * self.t.nq # list(best_result[:self.t.nq])
        self.ctrl_msg.effort = list(best_result[self.t.nx:(self.t.nx+self.t.nu)] + self.NOISE())
        self.publisher.publish(self.ctrl_msg)

        self.last_xs = self.xs
        self.last_u = np.array(self.ctrl_msg.effort)
        self.last_state_msg = msg
        # for i, result in enumerate(results):
        #     print(f"Result {i}: {result[:13]} ...")

        # self.last_joint_callback = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # self.get_logger().info(f"Publishing torques: {[round(t, 2) for t in torques]}")

def main(args=None):
    try:
        rclpy.init(args=args)
        torque_calculator = TorqueCalculator()
        rclpy.spin(torque_calculator)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        torque_calculator.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
