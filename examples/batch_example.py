#!/usr/bin/env python3
import numpy as np
import time
import sys
import os

np.set_printoptions(linewidth=1e8)

# Add the build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build'))

# Import the Python module
import pysqpcpu


def main():
    # Path to the URDF file
    urdf_filename = "/Users/emreadabag/code/indy-ros2/indy_description/urdf_files/indy7.urdf"
    
    # Parameters
    batch_size = 4
    N = 32
    dt = 0.01
    max_qp_iters = 5
    num_threads = 4  # Use 4 threads
    
    # Create a BatchThneed instance
    batch_solver = pysqpcpu.BatchThneed(
        urdf_filename=urdf_filename,
        batch_size=batch_size,
        N=N,
        dt=dt,
        max_qp_iters=max_qp_iters,
        num_threads=num_threads
    )
    
    # Create a single Thneed instance to get dimensions
    single_solver = pysqpcpu.Thneed(urdf_filename, N, dt, max_qp_iters)
    nx = single_solver.nx
    
    # Create batch inputs
    xs_batch = []
    eepos_g_batch = []
    
    # Initialize batch inputs with different values
    for i in range(batch_size):
        xs = np.zeros(nx)
        eepos_g = np.ones(3 * N)
        
        xs_batch.append(xs)
        eepos_g_batch.append(eepos_g)
    
    # Measure execution time
    start_time = time.time()
    
    # set xs for each solver
    batch_solver.batch_update_xs(xs_batch)

    # Run batch SQP
    batch_solver.batch_sqp(xs_batch, eepos_g_batch)
    
    # Calculate elapsed time
    elapsed = (time.time() - start_time) * 1000  # Convert to ms
    
    # Get results
    results = batch_solver.get_results()
    results_np = np.array(results)
    
    # Print results
    for i, result in enumerate(results):
        print(f"Result {i}: {result[:13]} ...")
    
    t = pysqpcpu.Thneed(urdf_filename, N, dt, max_qp_iters)
    t.setxs(xs_batch[0])
    t.sqp(xs_batch[0], eepos_g_batch[0])
    print(f"Truth: {t.XU[:13]}")
    
    print(f"Total execution time: {elapsed:.2f} ms")
    print(f"Average time per problem: {elapsed / batch_size:.2f} ms\n")
    
    
    ground_truth_dist = np.max(np.abs(t.XU - results_np.max(axis=0)))
    batch_dist = np.max(np.abs(results_np.max(axis=0) - results_np.min(axis=0)))
    
    print(f"Ground truth match: {ground_truth_dist < 1e-6}, {ground_truth_dist}")
    print(f"Batch consistency: {batch_dist < 1e-6}, {batch_dist}")
    
if __name__ == "__main__":
    main() 