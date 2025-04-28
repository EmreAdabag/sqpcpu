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

# set seed
np.random.seed(123)


def figure8():
    xamplitude = 0.4 # X goes from -xamplitude to xamplitude
    zamplitude = 0.8 # Z goes from -zamplitude/2 to zamplitude/2
    period = 5 # seconds
    dt = 0.01 # seconds
    x = lambda t:  0.5 + 0.1 * np.sin(2*(t + np.pi/4))
    y = lambda t: xamplitude * np.sin(t)
    z = lambda t: 0.1 + zamplitude * np.sin(2*t)/2 + zamplitude/2
    timesteps = np.linspace(0, 2*np.pi, int(period/dt))
    points = np.array([[x(t), y(t), z(t)] for t in timesteps]).reshape(-1)
    return points

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
        self.jointstate_count = 0
        
        
        

        urdf_filename = "urdfs/indy7.urdf"
        self.batch_size = 4
        self.num_threads = self.batch_size
        self.dt = 0.01
        self.fext_timesteps = 5
        N = 32
        max_qp_iters = 4
        num_threads = self.batch_size
        Q_cost = 2.0
        dQ_cost = 1e-3
        R_cost = 1e-6
        QN_cost = 10.0
        Qlim_cost = 0.00
        orient_cost = 0.0
        self.resample_fext = (self.batch_size > 1)
        self.usefext = False
        self.file_prefix = f'batchctrl_fext{self.batch_size}_linear2'

        self.config = {
            'file_prefix': self.file_prefix,
            'urdf_filename': urdf_filename,
            'batch_size': self.batch_size,
            'N': N,
            'dt': self.dt,
            'max_qp_iters': max_qp_iters,
            'num_threads': num_threads,
            'fext_timesteps': self.fext_timesteps,
            'Q_cost': Q_cost,
            'dQ_cost': dQ_cost,
            'R_cost': R_cost,
            'QN_cost': QN_cost,
            'Qlim_cost': Qlim_cost,
            'orient_cost': orient_cost,
            'resample_fext': self.resample_fext,
            'usefext': self.usefext,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }

        self.solver = pysqpcpu.BatchThneed(urdf_filename=urdf_filename, eepos_frame_name="end_effector", batch_size=self.batch_size, num_threads=self.num_threads, N=N, dt=self.dt, max_qp_iters=max_qp_iters, fext_timesteps=self.fext_timesteps, Q_cost=Q_cost, dQ_cost=dQ_cost, R_cost=R_cost, QN_cost=QN_cost, Qlim_cost=Qlim_cost, orient_cost=orient_cost)

        # self.solver.update_goal_orientation(np.array([[ 0., 1.,  0.],
        #                                               [ 0.,  0.,  1.],
        #                                               [1.,  0.,  0.]]))
        
        # facing right
        self.solver.update_goal_orientation(np.array([[ 0., -0.,  1.],
                                                [ 0.,  1.,  0.],
                                                [-1.,  0.,  0.]]))


        self.last_state_msg = None
        self.lastfrc = np.zeros(self.solver.nu) # true external forces (only first 3)

        self.fig8 = figure8()
        self.fig8_offset = 0
        self.goal_trace = self.fig8[:3*self.solver.N].copy()

        self.xs = np.zeros(self.solver.nx)
        self.eepos_g = np.zeros(3*self.solver.N)
        # self.fext_batch = 50.0 * np.ones((self.batch_size, 3))
        self.fext_batch = np.random.normal(0.0, 1.0, (self.batch_size, 6, self.solver.nq))
        self.fext_batch[0] = np.zeros((6, self.solver.nq))
        self.solver.batch_set_fext(self.fext_batch)
        self.last_xs = None
        self.last_u = None

    
        # stats
        self.tracking_errs = []
        self.positions = []

        self.last_joint_state_time = time.time()

    def getfext(self):
        if self.usefext:
            # self.lastfrc[2] =  20 * np.sin(self.jointstate_count * 0.005)
            self.lastfrc[2] =  10 * self.jointstate_count * 0.01
            self.lastfrc[1] = 10 * self.jointstate_count * 0.01
            return list(self.lastfrc)
        else:
            return [0.0] * self.solver.nu
    
    def goal_callback(self, msg):
        print('received pose: ', np.array(msg.pose.position))
        self.goal_trace = np.tile(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]), self.solver.N)
        self.eepos_g = self.goal_trace

    def joint_callback(self, msg):
        self.last_joint_state_time = time.time()
        self.jointstate_count += 1

        self.xs = np.hstack([np.array(msg.position), np.array(msg.velocity)])
        
        s = time.time()
        self.solver.sqp(self.xs, self.eepos_g) # run batch sqp
        print(f"batch sqp time: {1000 * (time.time() - s)} ms")

        
        
        if self.last_state_msg is not None:
            m1_time = self.last_state_msg.header.stamp.sec + self.last_state_msg.header.stamp.nanosec * 1e-9
            m2_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            step_duration = m2_time - m1_time
            # get prediction based on last applied control
            predictions = self.solver.predict_fwd(self.last_xs, self.last_u, step_duration)

            best_tracker_idx = None
            best_error = np.inf
            for i, result in enumerate(predictions):
                # get expected state for each result
                error = np.linalg.norm(result - self.xs)
                if error < best_error:
                    best_error = error
                    best_tracker_idx = i
            
            # resample fexts around the best result
            if self.resample_fext:
                self.fext_batch[:] = self.fext_batch[best_tracker_idx]
                self.fext_batch = np.random.normal(self.fext_batch, 0.10)
                self.fext_batch[:,:,3:] = 0.0
                self.solver.batch_set_fext(self.fext_batch)

        else:
            best_tracker_idx = 0

        print(f'most accurate force: {self.fext_batch[best_tracker_idx]}')
        
        best_result = self.solver.get_results()[best_tracker_idx]
        # Publish torques from batch result that best matched dynamics on the last step
        self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.ctrl_msg.position = [0.0] * self.solver.nq # list(best_result[:self.t.nq])
        self.ctrl_msg.velocity = self.getfext()
        self.ctrl_msg.effort = list(best_result[self.solver.nx:(self.solver.nx+self.solver.nu)])
        self.publisher.publish(self.ctrl_msg)

        self.last_xs = self.xs
        self.last_u = np.array(self.ctrl_msg.effort)
        self.last_state_msg = msg

        # record stats
        eepos = self.solver.eepos(self.xs[0:self.solver.nq])
        self.positions.append(eepos)
        self.tracking_errs.append(np.linalg.norm(eepos - self.goal_trace[:3]))
        if self.jointstate_count % 1000 == 0:
            # save tracking err to a file
            np.save(f'data/tracking_errs_{self.config["file_prefix"]}.npy', np.array(self.tracking_errs))
            np.save(f'data/positions_{self.config["file_prefix"]}.npy', np.array(self.positions))
            # shut down, ending all threads
            print('shutting down')
            rclpy.shutdown()
        # shift the goal trace
        self.goal_trace[:-3] = self.goal_trace[3:]
        self.goal_trace[-3:] = self.fig8[self.fig8_offset:self.fig8_offset+3]
        self.fig8_offset += 3
        self.fig8_offset %= len(self.fig8)
        self.eepos_g = self.goal_trace

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
