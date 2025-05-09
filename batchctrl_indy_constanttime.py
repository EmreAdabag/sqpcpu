import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from neuromeka import EtherCAT
import sys 
import os
current_dir = os.getcwd()
build_dir = os.path.join(current_dir, 'build')
sys.path.append(build_dir)
import pysqpcpu

# set seed
np.random.seed(123)

INDY = False


def figure8():
    xamplitude = 0.4 # X goes from -xamplitude to xamplitude
    zamplitude = 0.8 # Z goes from -zamplitude/2 to zamplitude/2
    period = 8 # seconds
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

        if INDY:
            self.ip = '160.39.102.88'
            self.ecat = EtherCAT(self.ip)
            self.servos = [
                {"index": 0, "direction": -1, "gear_ratio": 121, "ppr": 65536, "max_ecat_torque": 48.0, "rated_torque": 0.08839, "version":  "", "correction_rad": -0.054279739737023644, "torque_constant": 0.2228457},
                {"index": 1, "direction": -1, "gear_ratio": 121, "ppr": 65536, "max_ecat_torque": 48.0, "rated_torque": 0.0839705, "version":  "", "correction_rad": -0.013264502315156903, "torque_constant": 0.2228457},
                {"index": 2, "direction": 1, "gear_ratio": 121, "ppr": 65536, "max_ecat_torque": 96.0, "rated_torque": 0.0891443, "version":  "", "correction_rad": 2.794970264143719, "torque_constant": 0.10965625},
                {"index": 3, "direction": -1, "gear_ratio": 101, "ppr": 65536, "max_ecat_torque": 96.0, "rated_torque": 0.05798, "version":  "", "correction_rad": -0.0054105206811824215, "torque_constant": 0.061004},
                {"index": 4, "direction": -1, "gear_ratio": 101, "ppr": 65536, "max_ecat_torque": 96.0, "rated_torque": 0.055081, "version":  "", "correction_rad": 2.7930504019665254, "torque_constant": 0.061004},
                {"index": 5, "direction": -1, "gear_ratio": 101, "ppr": 65536, "max_ecat_torque": 96.0, "rated_torque": 0.05798, "version":  "", "correction_rad": -0.03490658503988659, "torque_constant": 0.061004}
            ]
            self.directions = np.array([self.servos[i]["direction"] for i in range(6)])
            self.torque_constants = np.array([self.servos[i]["torque_constant"] for i in range(6)])
            self.servo_min_torques = np.array([-431.97, -431.97, -197.23, -79.79, -79.79, -79.79], dtype=int)
            self.servo_max_torques = np.array([431.97, 431.97, 197.23, 79.79, 79.79, 79.79], dtype=int)
            
            # enable all servos
            self.ecat.set_servo(0, True)
            self.ecat.set_servo(1, True)
            self.ecat.set_servo(2, True)
            self.ecat.set_servo(3, True)
            self.ecat.set_servo(4, True)
            self.ecat.set_servo(5, True)

            self.run_period = 0.01  # 100Hz
            self.timer = self.create_timer(self.run_period, self.joint_callback)
        else:
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
        self.jointstate_count = 0
        
        
        print("USING TORQUE LIMITED INDY")
        urdf_filename = "urdfs/indy7_limited.urdf"
        self.batch_size = 1
        self.num_threads = self.batch_size
        self.dt = 0.01
        self.fext_timesteps = 5
        N = 32
        max_qp_iters = 4
        num_threads = self.batch_size
        Q_cost = 2.0
        dQ_cost = 2e-3
        R_cost = 2e-6
        QN_cost = 20.0
        Qpos_cost = 0.01
        Qvel_cost = 0.001
        Qacc_cost = 0.01
        orient_cost = 0.
        self.resample_fext = (self.batch_size > 1)
        self.usefext = False
        self.file_prefix = f'batchctrl_{self.batch_size}_{Qpos_cost}_{Qvel_cost}_{Qacc_cost}'

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
            'Qpos_cost': Qpos_cost,
            'Qvel_cost': Qvel_cost,
            'Qacc_cost': Qacc_cost,
            'orient_cost': orient_cost,
            'resample_fext': self.resample_fext,
            'usefext': self.usefext,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }

        self.solver = pysqpcpu.BatchThneed(urdf_filename=urdf_filename, eepos_frame_name="end_effector", 
                                            batch_size=self.batch_size, num_threads=self.num_threads, N=N, dt=self.dt, 
                                            max_qp_iters=max_qp_iters, fext_timesteps=self.fext_timesteps, Q_cost=Q_cost, 
                                            dQ_cost=dQ_cost, R_cost=R_cost, QN_cost=QN_cost, Qpos_cost=Qpos_cost, 
                                            Qvel_cost=Qvel_cost, Qacc_cost=Qacc_cost, orient_cost=orient_cost)

        # facing right
        # self.solver.update_goal_orientation(np.array([[ 0., -0.,  1.],
        #                                         [ 0.,  1.,  0.],
        #                                         [-1.,  0.,  0.]]))

        self.solver.update_goal_orientation(np.array([[ 0.  ,       -1.,          0.        ],
                                                    [ 0.99028227,  0.         , -0.13907202],
                                                    [ 0.13907202,  0.         ,  0.99028227]]))


        self.last_state_msg_time = None
        self.lastfrc = np.zeros(self.solver.nu) # true external forces (only first 3)'
        self.last_rotation = np.eye(3)

        self.fig8 = figure8()
        self.fig8_offset = 0
        self.goal_trace = self.fig8[:3*self.solver.N].copy()

        self.xs = np.zeros(self.solver.nx)
        self.eepos_g = self.fig8[:3*self.solver.N].copy()
        # self.eepos_g = np.tile(np.array([0., -.1865, 1.328]), self.solver.N)
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
        
    def set_normal_rotation(self, point):
        point[2] -= 0.3 # offset z to center around higher point
        # Normalize input vector (z-axis in new frame)
        z_axis = point / np.linalg.norm(point)

        # Find perpendicular vector (y-axis in new frame)
        ref = np.array([0., 1., 0.]) if not np.isclose(np.abs(z_axis.dot(np.array([0., 1., 0.]))), 1.0) else np.array([1., 0., 0.])
        y_axis = np.cross(z_axis, ref)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Find third perpendicular vector (x-axis in new frame)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Construct rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        self.last_rotation = 0.8 * rotation_matrix + 0.2 * self.last_rotation
        self.solver.update_goal_orientation(self.last_rotation)
    
    # def goal_callback(self, msg):
    #     print('received pose: ', np.array(msg.pose.position))
    #     self.goal_trace = np.tile(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]), self.solver.N)
    #     self.eepos_g = self.goal_trace

    def joint_callback(self, msg=None):
        self.last_joint_state_time = time.time()
        self.jointstate_count += 1

        if INDY:
            self.xs = np.zeros(self.solver.nx)
            for i in range(6):
                ppr = self.servos[i]["ppr"]
                gear_ratio = self.servos[i]["gear_ratio"]
                pos, vel, tor = self.ecat.get_servo_tx(i)[2:5]
                pos_rad = ((2 * math.pi * pos / gear_ratio / ppr) + self.servos[i]["correction_rad"]) * self.servos[i]["direction"]
                vel_rad = 2 * math.pi * vel / gear_ratio / ppr * self.servos[i]["direction"]
                self.xs[i] = pos_rad
                self.xs[i+6] = vel_rad
        else:
            self.xs = np.hstack([np.array(msg.position), np.array(msg.velocity)])

        # self.set_normal_rotation(self.eepos_g[3:6])
        print(f'xs: {self.xs.round(2)}')
        s = time.time()
        self.solver.sqp(self.xs, self.eepos_g) # run batch sqp
        # print(f"batch sqp time: {1000 * (time.time() - s)} ms")

        
        
        if self.last_state_msg_time is not None:
            m1_time = self.last_state_msg_time
            m2_time = time.monotonic()
            step_duration = m2_time - m1_time
            # get prediction based on last applied control
            predictions = self.solver.predict_fwd(self.last_xs, self.last_u, step_duration)

            best_tracker_idx = 0
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

        # print(f'most accurate force: {self.fext_batch[best_tracker_idx]}')
        
        best_result = self.solver.get_results()[best_tracker_idx]
        if INDY:
            torques_nm = 1.0 * best_result[self.solver.nx:self.solver.nx+6].clip(min=self.servo_min_torques, max=self.servo_max_torques) * self.directions
            servo_torques = np.round(torques_nm / self.torque_constants).astype(int)
            print(f'torques: {torques_nm.round(2)}')
            for i in range(6):
                self.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, servo_torques[i])
            self.last_u = torques_nm
        else:
            # Publish torques from batch result that best matched dynamics on the last step
            self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
            self.ctrl_msg.position = [0.0] * self.solver.nq # list(best_result[:self.t.nq])
            self.ctrl_msg.velocity = self.getfext()
            self.ctrl_msg.effort = list(best_result[self.solver.nx:(self.solver.nx+self.solver.nu)])
            self.publisher.publish(self.ctrl_msg)
            self.last_u = np.array(self.ctrl_msg.effort)
            print(f'torques: {self.ctrl_msg.effort}')

        self.last_xs = self.xs
        self.last_state_msg_time = time.monotonic()

        # record stats
        eepos = self.solver.eepos(self.xs[0:self.solver.nq])
        self.positions.append(eepos)
        self.tracking_errs.append(np.linalg.norm(eepos - self.goal_trace[:3]))
        if self.jointstate_count % 1000 == 0:
            # save tracking err to a file
            np.save(f'data/tracking_errs_{self.config["file_prefix"]}.npy', np.array(self.tracking_errs))
            np.save(f'data/positions_{self.config["file_prefix"]}.npy', np.array(self.positions))
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
        if INDY:
            for i in range(6):
                torque_calculator.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, 0)
            time.sleep(0.1)
            for i in range(6):
                torque_calculator.ecat.set_servo(i, False)
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
