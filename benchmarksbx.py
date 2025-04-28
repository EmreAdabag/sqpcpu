import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import sys 
import os
from tqdm import tqdm
import pickle
current_dir = os.getcwd()
build_dir = os.path.join(current_dir, 'build')
sys.path.append(build_dir)


BATCH = False
if BATCH:
    from pysqpcpu import BatchThneed
else:
    from pinocchio_template import Thneed


class Benchmark():
    def __init__(self, file_prefix='', batch_size=1, usefext=False):
        # xml_filename = "urdfs/frankapanda/mjx_panda.xml"
        urdf_filename = "urdfs/indy7.urdf"
        N = 32
        dt = 0.01
        max_qp_iters = 4
        num_threads = batch_size
        fext_timesteps = 8
        Q_cost = 0.0
        dQ_cost = 5e-3
        R_cost = 1e-6
        QN_cost = 0.0
        Qlim_cost = 0.00
        orient_cost = 0.1
        self.realtime = False
        self.resample_fext = (batch_size > 1)
        self.usefext = usefext
        self.file_prefix = file_prefix

        config = {
            'file_prefix': file_prefix,
            'urdf_filename': urdf_filename,
            'batch_size': batch_size,
            'N': N,
            'dt': dt,
            'max_qp_iters': max_qp_iters,
            'num_threads': num_threads,
            'fext_timesteps': fext_timesteps,
            'Q_cost': Q_cost,
            'dQ_cost': dQ_cost,
            'R_cost': R_cost,
            'QN_cost': QN_cost,
            'Qlim_cost': Qlim_cost,
            'orient_cost': orient_cost,
            'realtime': self.realtime,
            'resample_fext': self.resample_fext,
            'usefext': self.usefext,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        # pickle.dump(config, open(f'stats/{file_prefix}_benchmark_config.pkl', 'wb'))

        # solver
        if BATCH:
            self.solver = BatchThneed(urdf_filename=urdf_filename, eepos_frame_name="end_effector", batch_size=batch_size, num_threads=num_threads, N=N, dt=dt, max_qp_iters=max_qp_iters, fext_timesteps=fext_timesteps, Q_cost=Q_cost, dQ_cost=dQ_cost, R_cost=R_cost, QN_cost=QN_cost, Qlim_cost=Qlim_cost, orient_cost=orient_cost)
        else:
            self.solver = Thneed(urdf_filename=urdf_filename, eepos_frame_name="end_effector", N=N, dt=dt, max_qp_iters=max_qp_iters, Q_cost=Q_cost, dQ_cost=dQ_cost, R_cost=R_cost, QN_cost=QN_cost, Qlim_cost=Qlim_cost, orient_cost=orient_cost)

        # mujoco
        # self.model = mujoco.MjModel.from_xml_path(xml_filename)
        self.model = mujoco.MjModel.from_xml_path("urdfs/mujocomodels/indy7.xml")
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
    

        # points
        self.points = np.load('points/points1000.npy')[1:]
        # self.configs = np.load('points/configs10.npy')[1:]

        self.realfext_generator = np.random.default_rng(123)
        self.fext_generator = np.random.default_rng(321)

        # constants
        self.nq = self.solver.nq
        self.nv = self.solver.nv
        self.nu = self.solver.nu
        self.nx = self.solver.nx
        self.eepos_zero = self.solver.eepos(np.zeros(self.solver.nq))

        # tmp instance variables
        self.goal_trace = np.zeros(3*self.solver.N)
        self.xs = np.zeros(self.solver.nx)
        self.dist_to_goal = lambda goal_point: np.linalg.norm(self.solver.eepos(self.xs[:self.nq]) - goal_point[:3])
        self.last_control = np.zeros(self.solver.nu)
        self.fext_batch = self.fext_generator.normal(0.0, 10.0, (batch_size, 3))
        self.fext_batch[0] = np.array([0.0, 0.0, 0.0])
        if BATCH:
            self.solver.batch_set_fext(self.fext_batch)

        self.realfext = np.array([0.0, 0.0, -5.0]) * (self.usefext)

    
    def getfext(self):
        if self.usefext:
            self.realfext = np.clip(self.realfext_generator.normal(self.realfext, 2.0), -50.0, 50.0)
        return self.realfext
    
    def reset_solver(self):
        self.solver.reset_solver()

    def runMPC(self, viewer, goal_point):
        sim_steps = 0

        self.goal_trace = np.tile(goal_point[:3], self.solver.N)
        
        while sim_steps < 1000:
            if 0 and (self.dist_to_goal(goal_point) < 5e-2 and np.linalg.norm(self.data.qvel, ord=1) < 1.0):
                print(f'Got to goal in {sim_steps} steps')
                break

            # get current state
            self.xs = np.hstack((self.data.qpos, self.data.qvel))

            # solve
            solvestart = time.time()
            self.solver.sqp(self.xs, self.goal_trace)
            solve_time = time.time() - solvestart


            # simulate forward with last control
            sim_time = solve_time
            while sim_time > 0:
                
                self.data.xfrc_applied[7,:3] = self.getfext()
                # print(self.data.xfrc_applied[6,:3])
                mujoco.mj_step(self.model, self.data)
                sim_time -= self.model.opt.timestep
                
                if viewer is not None:
                    viewer.sync()
                sim_steps += 1
                
                # stats to print
                if sim_steps % 100 == 0:
                    linear_dist = self.dist_to_goal(goal_point)
                    # orientation_dist = np.linalg.norm(self.solver.compute_rotation_error(self.solver.eepos(self.xs[:self.nq])[1], self.solver.goal_orientation))
                    orientation_dist = 0
                    abs_vel = np.linalg.norm(self.data.qvel, ord=1)
                    print(np.round(linear_dist, 3), np.round(orientation_dist, 3), np.round(abs_vel, 3), np.round(1000 * (solve_time), 0), sep='\t')
                if not self.realtime:
                    break
            xnext = np.hstack((self.data.qpos, self.data.qvel))
            
            # get best control
            if BATCH:
                r = self.solver.get_results()
                predictions = self.solver.predict_fwd(self.xs, self.data.ctrl, self.model.opt.timestep * math.ceil(solve_time / self.model.opt.timestep))

                best_tracker = None
                best_error = np.inf
                for i, result in enumerate(predictions):
                    error = np.linalg.norm(result - xnext)
                    if error < best_error:
                        best_error = error
                        best_tracker = i
                # print(f'Best fext: {self.fext_batch[best_tracker]}')
                bestctrl = r[best_tracker][self.nx:self.nx+self.nu]

                if self.resample_fext:
                    self.fext_batch[:] = self.fext_batch[best_tracker]
                    self.fext_batch = self.fext_generator.normal(self.fext_batch, 2.0)
                    self.solver.batch_set_fext(self.fext_batch)
            else:
                bestctrl = self.solver.XU[self.nx:self.nx+self.nu]

            # set control for next step (maybe make this a moving avg so you don't give up gravity comp?)
            self.data.ctrl = bestctrl * 0.9 + self.last_control * 0.1
            # self.last_control = self.data.ctrl

    def runBench(self, headless=False):
        
        if headless:
            viewer = None
        else:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.solver.update_goal_orientation(np.array([[ 0., -0.,  1.],
                                                      [ 0.,  1.,  0.],
                                                      [-1.,  0.,  0.]]))
        

        # warmup
        for _ in range(10):
            self.solver.sqp(self.xs, self.goal_trace)
        
    
        # reset to zero
        self.xs = np.hstack((np.zeros(self.solver.nq), np.zeros(self.solver.nv)))
        self.data.qpos = self.xs[:self.solver.nq]
        self.data.qvel = self.xs[self.solver.nq:]
        self.data.ctrl = np.zeros(self.solver.nu)

        # go to point 1
        p1 = self.eepos_zero
        self.reset_solver()
        leg1 = self.runMPC(viewer, p1)

        if not headless:
            viewer.close()



        

if __name__ == '__main__':
    b = Benchmark(file_prefix='sbx_play', batch_size=1, usefext=False)
    b.runBench()