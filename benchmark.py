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
from pysqpcpu import Thneed, BatchThneed
# from pinocchio_template import Thneed

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=1000)

BATCH = True

class Benchmark():
    def __init__(self, file_prefix='', batch_size=1, N=16, usefext=False):
        # xml_filename = "urdfs/frankapanda/mjx_panda.xml"
        urdf_filename = "urdfs/indy7.urdf"
        dt = 0.01
        max_qp_iters = 5
        num_threads = batch_size
        fext_timesteps = 0
        Q_cost = 2.0
        dQ_cost = 1e-2
        R_cost = 1e-6
        QN_cost = 20.0
        Qpos_cost = 0.01
        Qvel_cost = 0.0 # don't unzero
        Qacc_cost = 0.01
        orient_cost = 0.0
        self.realtime = True
        self.resample_fext = 0 and (batch_size > 1)
        self.usefext = usefext
        self.file_prefix = file_prefix


        print(f"REALTIME: {self.realtime}")

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
            'Qpos_cost': Qpos_cost,
            'Qvel_cost': Qvel_cost,
            'Qacc_cost': Qacc_cost,
            'orient_cost': orient_cost,
            'realtime': self.realtime,
            'resample_fext': self.resample_fext,
            'usefext': self.usefext,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        # pickle.dump(config, open(f'stats/{file_prefix}_benchmark_config.pkl', 'wb'))

        # solver
        if BATCH:
            self.solver = BatchThneed(urdf_filename=urdf_filename, eepos_frame_name="link6", batch_size=batch_size, num_threads=num_threads, N=N, dt=dt, max_qp_iters=max_qp_iters, fext_timesteps=fext_timesteps, Q_cost=Q_cost, dQ_cost=dQ_cost, R_cost=R_cost, QN_cost=QN_cost, Qpos_cost=Qpos_cost, Qvel_cost=Qvel_cost, Qacc_cost=Qacc_cost, orient_cost=orient_cost)
        else:
            self.solver = Thneed(urdf_filename=urdf_filename, eepos_frame_name="end_effector", N=N, dt=dt, max_qp_iters=max_qp_iters, Q_cost=Q_cost, dQ_cost=dQ_cost, R_cost=R_cost, QN_cost=QN_cost, poslim_cost=poslim_cost, vellim_cost=vellim_cost, acclim_cost=acclim_cost, orient_cost=orient_cost)

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
        self.fext_mask = np.zeros((batch_size, 6, self.solver.nq+1))
        self.fext_mask[:,0,self.solver.nq] = 1
        self.fext_mask[:,1,self.solver.nq] = 1
        self.fext_mask[:,2,self.solver.nq] = 1
        self.fext_batch = np.zeros_like(self.fext_mask)
        if self.usefext:
            self.solver.batch_set_fext(self.fext_batch)

        self.realfext = np.array([0.0, 0.0, -5.0]) * (self.usefext)
    
    def getfext(self):
        if self.usefext:
            self.realfext = np.clip(self.realfext_generator.normal(self.realfext, 2.0), -50.0, 50.0)
        return self.realfext
    
    def reset_solver(self):
        # reset primals and duals to zeros
        # reset fexts to zeros
        # self.solver.batch_reset_solvers()
        # self.solver.clean_start()
        self.solver.reset_solver()

    def runMPC(self, viewer, goal_point):
        sim_steps = 0
        solves = 0
        total_cost = 0
        total_dist = 0
        total_ctrl = 0.0
        total_vel = 0.0
        best_cost = np.inf
        avg_solve_time = 0
        max_solve_time = 0

        self.goal_trace = np.tile(self.eepos_zero, self.solver.N)
        goal_set = False
        
        while sim_steps < 2000:
            if (self.dist_to_goal(goal_point) < 5e-2 and np.linalg.norm(self.data.qvel, ord=1) < 1.0):
                print(f'Got to goal in {sim_steps} steps')
                break

            # set goal
            if not goal_set and self.dist_to_goal(self.eepos_zero) < 0.4:
                self.goal_trace = np.tile(goal_point[:3], self.solver.N)
                goal_set = True

            # get current state
            self.xs = np.hstack((self.data.qpos, self.data.qvel))

            # solve
            solvestart = time.monotonic()
            self.solver.sqp(self.xs, self.goal_trace)
            solve_time = time.monotonic() - solvestart
            # print(f"Solve time: {1000 * solve_time} ms")

            # simulate forward with last control
            sim_time = solve_time
            while sim_time > 0:
                total_ctrl += np.linalg.norm(self.data.ctrl)
                total_vel += np.linalg.norm(self.data.qvel, ord=1)
                total_cost += self.solver.eepos_cost(np.hstack((self.data.qpos, self.data.qvel)), self.goal_trace, 1)
                total_dist += self.dist_to_goal(self.goal_trace[:3])
                
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
                # print(np.array(r)[:,12:18])
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
            self.data.ctrl = bestctrl * 0.85 # + self.last_control * 0.2
            # self.last_control = self.data.ctrl

            # update stats
            avg_solve_time += solve_time
            max_solve_time = max(max_solve_time, solve_time)
            best_cost = min(best_cost, self.solver.last_state_cost)
            solves += 1

        stats = {
            'failed': sim_steps>=1000,
            'cumulative_dist': total_dist,
            'cumulative_cost': total_cost,
            'best_cost': best_cost,
            'avg_solve_time': avg_solve_time / solves,
            'max_solve_time': max_solve_time,
            'steps': sim_steps,
            'solves': solves,
            'total_ctrl': total_ctrl,
            'total_vel': total_vel
        }
        print(f'average vel: {total_vel / sim_steps}')
        print(f'average ctrl: {total_ctrl / sim_steps}')
        return stats

    def runBench(self, headless=True):

        allstats = {
            'failed': [],
            'cumulative_cost': [],
            'cumulative_dist': [],
            'best_cost': [],
            'avg_solve_time': [],
            'max_solve_time': [],
            'steps': [],
            'total_ctrl': [],
            'total_vel': []
        }
        
        if headless:
            viewer = None
        else:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        
        

        # warmup
        # for _ in range(10):
        #     self.solver.sqp(self.xs, self.goal_trace)
        
        for i in tqdm(range(len(self.points)-1)):
            print(f'Point{i}: {self.points[i]}, {self.points[i+1]}')
            # reset to zero
            self.xs = np.hstack((np.zeros(self.solver.nq), np.zeros(self.solver.nv)))
            self.data.qpos = self.xs[:self.solver.nq]
            self.data.qvel = self.xs[self.solver.nq:]
            self.data.ctrl = np.zeros(self.solver.nu)

            # go to point 1
            p1 = self.points[i]
            self.reset_solver()
            leg1 = self.runMPC(viewer, p1)

            # go to point 2
            p2 = self.points[i+1]
            self.reset_solver()
            leg2 = self.runMPC(viewer, p2)

            # return to zero
            self.reset_solver()
            leg3 = self.runMPC(viewer, self.eepos_zero)

            failed = leg1['failed'] or leg2['failed'] or leg3['failed']
            print(f'Failed: {failed}')
            allstats['failed'].append(failed)
            allstats['cumulative_cost'].append(leg1['cumulative_cost'] + leg2['cumulative_cost'] + leg3['cumulative_cost'])
            allstats['cumulative_dist'].append(leg1['cumulative_dist'] + leg2['cumulative_dist'] + leg3['cumulative_dist'])
            allstats['best_cost'].append(min(leg1['best_cost'], leg2['best_cost'], leg3['best_cost']))
            allstats['avg_solve_time'].append((leg1['avg_solve_time'] * leg1['steps'] + leg2['avg_solve_time'] * leg2['steps'] + leg3['avg_solve_time'] * leg3['steps']) / (leg1['steps'] + leg2['steps'] + leg3['steps']))
            allstats['max_solve_time'].append(max(leg1['max_solve_time'], leg2['max_solve_time'], leg3['max_solve_time']))
            allstats['steps'].append(leg1['steps'] + leg2['steps'] + leg3['steps'])
            
            # save stats
            # if i % 20 == 0:
            #     pickle.dump(allstats, open(f'stats/{self.file_prefix}_stats_{i}.pkl', 'wb'))
            if i==50:
                break
        pickle.dump(allstats, open(f'stats/{self.file_prefix}_stats_final.pkl', 'wb'))

        if not headless:
            viewer.close()

        return allstats


        

if __name__ == '__main__':

    combos = [(8,32), (1,64)]
    for batch_size, N in combos:
        b = Benchmark(file_prefix=f'stockcpu_batch{batch_size}_{N}', batch_size=batch_size, N=N, usefext=False)
        b.runBench()