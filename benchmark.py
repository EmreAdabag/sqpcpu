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
# import pysqpcpu
from pinocchio_template import thneed


class Benchmark():
    def __init__(self):
        # xml_filename = "urdfs/frankapanda/mjx_panda.xml"
        urdf_filename = "urdfs/indy7.urdf"
        batch_size = 1
        N = 8
        dt = 0.01
        max_qp_iters = 2
        num_threads = 1
        fext_timesteps = 0
        Q_cost = 1.0
        dQ_cost = 5e-3
        R_cost = 1e-6
        QN_cost = 1.0
        Qlim_cost = 0
        orient_cost = 0.01

        # solver
        # self.solver = pysqpcpu.BatchThneed(xml_filename=xml_filename, eepos_frame_name="hand", batch_size=batch_size, N=N, dt=dt, max_qp_iters=max_qp_iters, num_threads=num_threads, fext_timesteps=fext_timesteps, dQ_cost=dQ_cost, R_cost=R_cost, QN_cost=QN_cost, Qlim_cost=Qlim_cost, regularize_cost=regularize_cost, discount_factor=discount_factor)
        self.solver = thneed(urdf_filename=urdf_filename, eepos_frame_name="end_effector", N=N, dt=dt, max_qp_iters=max_qp_iters, dQ_cost=dQ_cost, R_cost=R_cost, QN_cost=QN_cost, Q_cost=Q_cost, Qlim_cost=Qlim_cost, orient_cost=orient_cost)

        # mujoco
        # self.model = mujoco.MjModel.from_xml_path(xml_filename)
        self.model = mujoco.MjModel.from_xml_path("urdfs/mujocomodels/indy7.xml")
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
    

        # points
        self.points = np.load('points/points1000.npy')[1:]
        # self.configs = np.load('points/configs10.npy')[1:]

        # constants
        self.nq = self.solver.nq
        self.nv = self.solver.nv
        self.nu = self.solver.nu
        self.nx = self.solver.nx

        # tmp instance variables
        self.goal_trace = np.zeros(3*self.solver.N)
        self.xs = np.zeros(self.solver.nx)
        self.dist_to_goal = lambda: np.linalg.norm(self.solver.eepos(self.xs[:self.nq])[0] - self.goal_trace[-3:])

        self.last_control = np.zeros(self.solver.nu)
    
    def reset_solver(self):
        # reset primals and duals to zeros
        # reset fexts to zeros
        # self.solver.batch_reset_solvers()
        # self.solver.clean_start()
        self.solver.reset_solver()

    def runMPC(self, viewer):
        # assume self.xs and self.goal_trace are set
        sim_steps = 0
        solves = 0
        cumulative_cost = 0
        best_cost = np.inf
        avg_solve_time = 0
        max_solve_time = 0
        
        while sim_steps < 1000:
            if (self.dist_to_goal() < 5e-2 and np.linalg.norm(self.data.qvel, ord=1) < 1.0):
                print(f'Got to goal in {sim_steps} steps')
                break

            # get current state
            self.xs = np.hstack((self.data.qpos, self.data.qvel))

            # solve
            solvestart = time.time()
            self.solver.setxs(self.xs)
            self.solver.sqp(self.xs, self.goal_trace)
            # r = self.solver.get_results()
            solve_time = time.time() - solvestart

            # get best control
            bestctrl = self.solver.XU[self.nx:self.nx+self.nu]
            
            # simulate forward
            sim_time = solve_time
            while sim_time > 0:
                # step fwd with last control
                mujoco.mj_step(self.model, self.data)
                sim_time -= self.model.opt.timestep
                
                if viewer is not None:
                    viewer.sync()
                sim_steps += 1
                
                # stats to print
                if sim_steps % 100 == 0:
                    linear_dist = self.dist_to_goal()
                    orientation_dist = np.linalg.norm(self.solver.compute_rotation_error(self.solver.eepos(self.xs[:self.nq])[1], self.solver.goal_orientation))
                    abs_vel = np.linalg.norm(self.data.qvel, ord=1)
                    print(np.round(linear_dist, 3), np.round(orientation_dist, 3), np.round(abs_vel, 3), np.round(1000 * (solve_time), 0), sep='\t')

            # set control for next step
            self.data.ctrl = bestctrl[:] * 0.75 + self.last_control[:] * 0.25

            # update stats
            avg_solve_time += solve_time
            max_solve_time = max(max_solve_time, solve_time)
            qc, vc, uc = self.solver.eepos_cost(self.goal_trace, self.xs, 1)
            cumulative_cost += qc + vc + uc
            best_cost = min(best_cost, cumulative_cost)
            solves += 1
        
        stats = {
            'failed': sim_steps>=1000,
            'cumulative_cost': cumulative_cost,
            'best_cost': best_cost,
            'avg_solve_time': avg_solve_time / solves,
            'max_solve_time': max_solve_time,
            'steps': sim_steps,
            'solves': solves
        }

        return stats

    def runBench(self, headless=False):

        allstats = {
            'failed': [],
            'cumulative_cost': [],
            'best_cost': [],
            'avg_solve_time': [],
            'max_solve_time': [],
            'steps': []
        }
        
        if headless:
            viewer = None
        else:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        
        eepos_zero = self.solver.eepos(np.zeros(self.solver.nq))[0]
        
        
        for i in tqdm(range(len(self.points)-1)):
            if i < 19:
                continue
            print(f'Point{i}: {self.points[i]}, {self.points[i+1]}')
            # reset to zero
            self.xs = np.hstack((np.zeros(self.solver.nq), np.zeros(self.solver.nv)))
            self.data.qpos = self.xs[:self.solver.nq]
            self.data.qvel = self.xs[self.solver.nq:]
            self.data.ctrl = np.zeros(self.solver.nu)

            # go to point 1
            # p1 = self.points[i]
            # self.reset_solver()
            # self.goal_trace = np.tile(p1, self.solver.N)
            # leg1 = self.runMPC(viewer)

            # go to point 2
            p2 = self.points[i+1]
            self.reset_solver()
            self.goal_trace = np.tile(p2, self.solver.N)
            leg2 = self.runMPC(viewer)
            exit(1)

            # return to zero
            self.reset_solver()
            self.goal_trace = np.tile(eepos_zero, self.solver.N)
            leg3 = self.runMPC(viewer)

            failed = leg1['failed'] or leg2['failed'] or leg3['failed']
            print(f'Failed: {failed}')
            allstats['failed'].append(failed)
            allstats['cumulative_cost'].append(leg1['cumulative_cost'] + leg2['cumulative_cost'] + leg3['cumulative_cost'])
            allstats['best_cost'].append(min(leg1['best_cost'], leg2['best_cost'], leg3['best_cost']))
            allstats['avg_solve_time'].append((leg1['avg_solve_time'] * leg1['steps'] + leg2['avg_solve_time'] * leg2['steps'] + leg3['avg_solve_time'] * leg3['steps']) / (leg1['steps'] + leg2['steps'] + leg3['steps']))
            allstats['max_solve_time'].append(max(leg1['max_solve_time'], leg2['max_solve_time'], leg3['max_solve_time']))
            allstats['steps'].append(leg1['steps'] + leg2['steps'] + leg3['steps'])
            
            # save stats
            if i % 100 == 0:
                pickle.dump(allstats, open(f'stats/stats_{i}.pkl', 'wb'))
        pickle.dump(allstats, open(f'stats/stats_final.pkl', 'wb'))

        if not headless:
            viewer.close()

        return allstats


        

if __name__ == '__main__':
    b = Benchmark()
    b.runBench()